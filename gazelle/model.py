import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV3Backbone

'''
模块 2: GGSF (Geometry-Guided Spatial Focus)
功能： 根据人头位置，生成一个“注意力Mask”，抑制背景噪声。
原理： 乘法门控 (Multiplicative Gating)。
'''
class GeometryGuidedSpatialFocus(nn.Module):
    """
    Contribution 2: GGSF
    利用人头位置构建主动几何场，通过乘性门控抑制非视域区域的噪声。
    """
    def __init__(self, feat_h, feat_w, embed_dim):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        
        # 输入是相对坐标 (dx, dy)，2通道
        # 输出是 1 通道的 Attention Mask
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() # 输出 0~1 的权重
        )

    def forward(self, bboxes, num_ppl_per_img, device):
        # 1. 构建几何坐标场 (Geometric Coordinate Field)
        # 我们生成一个与特征图同大小的网格，计算每个像素相对于人头中心的偏移量
        
        batch_geo_maps = []
        
        # 这里的逻辑稍微复杂一点，因为 bboxes 是 list of list
        # 我们需要在 "Repeat Features" 之后，也就是对每个人都生成一个 map
        # 为了高效，我们先按图片生成，再根据 num_ppl_per_img 拆分或重组
        
        # 构造网格 (1, 2, H, W)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.feat_h, device=device),
            torch.arange(self.feat_w, device=device),
            indexing='ij'
        )
        # 归一化到 0-1
        y_grid = y_grid.float() / self.feat_h
        x_grid = x_grid.float() / self.feat_w
        base_grid = torch.stack([x_grid, y_grid], dim=0) # [2, H, W]

        # 遍历每个 Bbox 生成对应的相对坐标图
        final_geo_masks = []
        
        bbox_idx = 0
        for i, bbox_list in enumerate(bboxes):
            for bbox in bbox_list:
                if bbox is None:
                     # 异常处理
                    cx, cy = 0.5, 0.5
                else:
                    # 计算 Bbox 中心点
                    xmin, ymin, xmax, ymax = bbox
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                
                # 计算相对坐标 (dx, dy)
                # grid: [2, H, W] - center: [2, 1, 1]
                center = torch.tensor([cx, cy], device=device).view(2, 1, 1)
                relative_coords = base_grid - center # 此时原点在人头中心
                
                final_geo_masks.append(relative_coords)
        
        if len(final_geo_masks) == 0:
            return None

        # 堆叠所有人的几何图: [Total_People, 2, H, W]
        geo_input = torch.stack(final_geo_masks)
        
        # 2. 通过 MLP 生成空间门控掩码 (Spatial Gating Mask)
        attention_mask = self.geo_mlp(geo_input) # [Total_People, 1, H, W]
        
        return attention_mask

'''
模块 1: SASA (Scale-Aware Semantic Aggregation)
功能： 动态选择 4 层特征的权重，并融合。
原理： Attention (Query=HeadToken, Key=FeatureGlobalAvg)。
'''
class ScaleAwareSemanticAggregator(nn.Module):
    """
    Contribution 1: SASA
    动态层级选择模块。利用 Head Token 作为 Query，自适应地融合多层特征。
    """
    def __init__(self, in_dim, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.in_dim = in_dim
        
        # 计算权重的注意力模块
        # Input: Head Token (dim) + Feature Global Pool (dim)
        self.scale_attention = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1), # 输出 4 个权重值
            nn.Softmax(dim=1) 
        )
        
        # 融合后的降维 (如果需要) 或 特征变换
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

    def forward(self, features_list, head_token):
        # features_list: List of [B_people, C, H, W] (共4个)
        # head_token: [1, C] or [B_people, C]
        
        # 1. 堆叠特征: [B_people, 4, C, H, W]
        stacked_feats = torch.stack(features_list, dim=1) 
        b, n, c, h, w = stacked_feats.shape
        
        # 2. 提取每层的全局语义向量 (Global Average Pooling)
        # [B, 4, C, H, W] -> [B, 4, C]
        feats_global = torch.mean(stacked_feats, dim=[3, 4]) 
        
        # 3. 计算 Head Token 与每层特征的相关性
        # head_token 扩充: [B, C] -> [B, 1, C]
        if head_token.dim() == 2:
            query = head_token.unsqueeze(1)
        else:
            # 兼容 Parameter 情况
            query = head_token.view(1, 1, -1).repeat(b, 1, 1)

        # 简单的加法融合或者是 Concat 融合来算 Attention
        # 这里我们把 query 加到每一层特征上，算一个分数
        # [B, 4, C] + [B, 1, C] -> [B, 4, C] -> MLP -> [B, 4]
        fusion_for_attn = feats_global + query 
        
        # 展平做 MLP: [B*4, C] (为了简单实现，或者用循环)
        # 这里直接用 Linear 处理最后一维 C
        weights = self.scale_attention(fusion_for_attn) # [B, 4]
        
        # 4. 加权融合
        # weights: [B, 4] -> [B, 4, 1, 1, 1]
        weights = weights.view(b, n, 1, 1, 1)
        
        # Weighted Sum: sum([B, 4, C, H, W] * weights) -> [B, C, H, W]
        fused_feat = torch.sum(stacked_feats * weights, dim=1)
        
        # 5. 投影变换
        out = self.project(fused_feat)
        
        return out, weights # 返回 weights 可用于可视化分析


class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(512, 512), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        # 我们需要一个降维层，把 DINO 的 1024 降到 256
        self.feat_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.get_dimension(), self.dim, 1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            ) for _ in range(4) # 对4层特征都做适配
        ])
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # Contribution 1: SASA Module
        self.sasa = ScaleAwareSemanticAggregator(self.dim, num_scales=4)
        # Contribution 2: GGSF Module
        self.ggsf = GeometryGuidedSpatialFocus(self.featmap_h, self.featmap_w, self.dim)
        # Contribution 3: Aux Head (辅助监督),用于最浅层 (Layer 4) 的监督
        self.aux_head = nn.Sequential(
            nn.ConvTranspose2d(self.dim, self.dim, kernel_size=2, stride=2),
            nn.Conv2d(self.dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.head_token = nn.Embedding(1, self.dim)
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

        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        total_people = sum(num_ppl_per_img)
        if total_people == 0:
            # 边界情况处理...
            pass
        raw_features_list = self.backbone.forward(input["images"])
        adapted_features = []
        for i, feat in enumerate(raw_features_list):
            # [B_img, 1024, H, W] -> [B_img, 256, H, W]
            feat = self.feat_adapter[i](feat)
            # [B_img, 256, H, W] -> [Total_People, 256, H, W]
            feat = utils.repeat_tensors(feat, num_ppl_per_img)
            adapted_features.append(feat)

        # Contribution 2: GGSF (几何门控) 生成每个人的几何掩码: [Total_People, 1, H, W]
        geo_mask = self.ggsf(input["bboxes"], num_ppl_per_img, adapted_features[0].device)
        # 将几何掩码乘到每一层特征上 (Multiplicative Gating) 这一步体现了“物理约束”：抑制了视野外的特征
        gated_features = []
        for feat in adapted_features:
            gated_features.append(feat * geo_mask) # Broadcast multiply

        # Contribution 3: Aux Head (辅助监督) - 针对最浅层,我们取最浅层 (index 0, 对应 Layer 4) 
        shallow_feat = gated_features[0] 
        # 简单处理后预测
        aux_out = self.aux_head(shallow_feat) # [Total_People, 1, H*2, W*2]
        # Resize 到输出大小
        aux_out = torchvision.transforms.functional.resize(aux_out, self.out_size).squeeze(1)
        aux_preds = utils.split_tensors(aux_out, num_ppl_per_img)

        # Contribution 1: SASA (特征融合),使用 Head Token 动态融合 4 层特征
        # gated_features: List of [Total_People, 256, H, W]
        # self.head_token.weight: [1, 256]
        x, layer_weights = self.sasa(gated_features, self.head_token.weight)
        # x: [Total_People, 256, H, W]

        x = x + self.pos_embed
        # Gazelle 原有的 Head Map Embedding (加法)
        # 既然我们用了 GGSF (乘法)，这里的加法可以保留作为补充，或者去掉,建议保留，双重保障 (Hybrid Fusion)
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        
        x = x.flatten(start_dim=2).permute(0, 2, 1) # [B, HW, C]

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] # slice off inout tokens from scene tokens
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2) # b (h w) c -> b c h w
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img) # resplit per image

        return {
            "heatmap": heatmap_preds, 
            "aux_heatmap": aux_preds, # 返回这个用于计算辅助 Loss
            "inout": inout_preds if self.inout else None,
            "layer_weights": layer_weights # 可视化用
        }

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
