import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV3Backbone

import torch.nn.functional as F # 新增，用于归一化
import sys, os
sys.path.append('/home/fb/src/paper/gazelleV1/dinov3') 
# 我们需要从 DINOv3 的 hub 中加载配置好的文本模型
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

backbone_path = "/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth.pth"
dinotxt_path = "/home/fb/src/paper/gazelleV1/checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"

class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(512, 512), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # === 1. 核心修改：加载文本编码器 ===
        # 这里我们临时加载一个完整的 DINOTxt 模型，只取它的 text_model
        # 注意：你需要确保下载了对应的权重，或者让其自动下载
        print("Loading Text Encoder from DINOv3...")
        if not os.path.exists(backbone_path) or not os.path.exists(dinotxt_path):
            raise FileNotFoundError(f"请确保以下两个文件都在 checkpoints/ 目录下:\n1. {backbone_path}\n2. {dinotxt_path}")
        # full_dinotxt, self.tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True)
        full_dinotxt, self.tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
            pretrained=True,
            weights=dinotxt_path,           # <--- 传入本地 Text/Align 权重路径
            backbone_weights=backbone_path  # <--- 传入本地 Backbone 权重路径
        )
        self.text_backbone = full_dinotxt.text_model
        # 冻结文本编码器（通常建议先冻结，只训练解码器）
        for param in self.text_backbone.parameters():
            param.requires_grad = False
        
        # 我们需要一个投影层，把文本特征映射到 GazeLLE 的维度 (dim=256)
        self.text_proj = nn.Linear(2048, self.dim) 
        # =================================

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)

        # === 2. 修改：不再需要 head_token ===
        # 原来的 head_token 是为了给 BBox 区域加 embedding
        # 现在我们要用文本生成的 mask，所以这个可以保留作为 learnable scale，或者直接去掉
        # 建议保留一个可学习的缩放系数，控制文本提示的强度
        self.prompt_scale = nn.Parameter(torch.ones(1) * 0.1) 
        # ==================================        
        
        # self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        if self.inout: self.inout_token = nn.Embedding(1, self.dim)
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1)
                for i in range(num_layers)
                ])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.inout: 
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, input):
        # input["images"]: [B, 3, H, W] tensor of images
        # input["bboxes"]: list of lists of bbox tuples [[(xmin, ymin, xmax, ymax)]] per image in normalized image coords
        # input["text"]: list of strings, e.g. ["man in red", "woman in blue"]

        # 1. 提取图像特征
        x_vis = self.backbone.forward(input["images"]) # [B, C_vis, H, W]
        # 2. 提取文本特征
        # 需要将文本 tokenize 并放入 device
        text_tokens = self.tokenizer.tokenize(input["text"]).to(x_vis.device)
        text_emb = self.text_backbone(text_tokens) # [B, 1280] (假设是 CLS token 或 pooled output)
        text_emb = F.normalize(text_emb, dim=-1) # 归一化很重要
        # 3. 生成“语言引导 Mask” (Language-Guided Mask)
        # 我们计算文本特征与图像空间特征的点积相似度
        # DINOv3 的视觉特征 x_vis 需要先归一化以便计算余弦相似度
        B, C, H, W = x_vis.shape
        x_vis_norm = F.normalize(x_vis, dim=1) # 在通道维度归一化
        # === 简化版实现逻辑 ===
        x = self.linear(x_vis) # [B, 256, H, W] - 图像特征降维到 GazeLLE 维度
        x = x + self.pos_embed
        # 将文本也映射到 256 维
        text_emb_proj = self.text_proj(text_emb).unsqueeze(-1).unsqueeze(-1) # [B, 256, 1, 1]
        # 计算 Attention Mask: 图像特征 与 文本特征 的点积
        # 形状: [B, 256, H, W] * [B, 256, 1, 1] -> sum -> [B, 1, H, W]
        prompt_mask = (x * text_emb_proj).sum(dim=1, keepdim=True) 
        
        # 激活一下，让 Mask 更关注正样本区域（类似 ReLU 或 Sigmoid）
        # 或者直接作为一种 bias 加进去
        # 这里模拟原论文的 add embedding 操作
        x = x + (prompt_mask * self.prompt_scale)

        # num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        # x = self.backbone.forward(input["images"])
        # x = self.linear(x)
        # x = x + self.pos_embed
        # x = utils.repeat_tensors(x, num_ppl_per_img) # repeat image features along people dimension per image
        # head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device) # [sum(N_p), 32, 32]
        # head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        # x = x + head_map_embeddings
        x = x.flatten(start_dim=2).permute(0, 2, 1) # "b c h w -> b (h w) c"

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        # 注意：现在不需要 repeat_tensors 和 split_tensors 了，
        # 因为我们假设一个文本对应生成一张热力图，也就是 batch size 是一一对应的。
        # 如果一张图有多个文本 query，你应该在 dataloader 阶段把图片复制多次形成 batch。
        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            # inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] # slice off inout tokens from scene tokens
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        # heatmap_preds = utils.split_tensors(x, num_ppl_per_img) # resplit per image

        return {"heatmap": x, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: # no bbox provided, use empty head map
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps
    
    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}
        
    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        
        self.load_state_dict(current_state_dict, strict=False)


# From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
    

# models
def get_gazelle_model(model_name):
    factory = {
        "gazelle_dinov3_vitb16": gazelle_dinov3_vitb16,
        "gazelle_dinov3_vitl16": gazelle_dinov3_vitl16,
        "gazelle_dinov3_vitb16_inout": gazelle_dinov3_vitb16_inout,
        "gazelle_dinov3_vitl16_inout": gazelle_dinov3_vitl16_inout,
    }
    assert model_name in factory.keys(), "invalid model name"
    return factory[model_name]()

def gazelle_dinov3_vitb16():
    backbone = DinoV3Backbone('dinov3_vitb16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov3_vitl16():
    backbone = DinoV3Backbone('dinov3_vitl16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov3_vitb16_inout():
    backbone = DinoV3Backbone('dinov3_vitb16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, inout=True)
    return model, transform

def gazelle_dinov3_vitl16_inout():
    backbone = DinoV3Backbone('dinov3_vitl16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, inout=True)
    return model, transform
