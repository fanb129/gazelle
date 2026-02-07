import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import torch.nn.functional as F

import gazelle.utils as utils
from gazelle.backbone import DinoV3Backbone

# ==========================================
# 模块 1: GGSF (Geometry-Guided Spatial Focus)
# ==========================================
class GeometryGuidedSpatialFocus(nn.Module):
    def __init__(self, feat_h, feat_w):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        
        # 输入是相对坐标 (dx, dy)，2通道
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, bboxes, num_ppl_per_img, device):
        # 构造网格 (1, 2, H, W)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.feat_h, device=device),
            torch.arange(self.feat_w, device=device),
            indexing='ij'
        )
        y_grid = y_grid.float() / self.feat_h
        x_grid = x_grid.float() / self.feat_w
        base_grid = torch.stack([x_grid, y_grid], dim=0) # [2, H, W]

        final_geo_masks = []
        for i, bbox_list in enumerate(bboxes):
            for bbox in bbox_list:
                if bbox is None:
                    cx, cy = 0.5, 0.5
                else:
                    xmin, ymin, xmax, ymax = bbox
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                
                center = torch.tensor([cx, cy], device=device).view(2, 1, 1)
                relative_coords = base_grid - center 
                final_geo_masks.append(relative_coords)
        
        if len(final_geo_masks) == 0:
            return None

        geo_input = torch.stack(final_geo_masks) # [Total_People, 2, H, W]
        attention_mask = self.geo_mlp(geo_input) # [Total_People, 1, H, W]
        
        return attention_mask

# ==========================================
# 模块 2: SASA (Scale-Aware Semantic Aggregation)
# ==========================================
class ScaleAwareSemanticAggregator(nn.Module):
    def __init__(self, in_dim, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        
        # 计算权重: Input [Head_Token + Global_Pool]
        self.scale_attention = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_scales),
            nn.Softmax(dim=1) 
        )
        
        # 融合后的特征变换
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

    def forward(self, features_list, head_token):
        # features_list: List of [Total_People, C, H, W]
        stacked_feats = torch.stack(features_list, dim=1) # [B, 4, C, H, W]
        b, n, c, h, w = stacked_feats.shape
        
        # Global Pool for Attention
        feats_global = torch.mean(stacked_feats, dim=[3, 4]) # [B, 4, C]
        
        if head_token.dim() == 2:
            query = head_token.unsqueeze(1) # [B, 1, C]
        else:
            query = head_token.view(1, 1, -1).repeat(b, 1, 1)

        # 简单的相加融合用于计算 Attention Score
        fusion_for_attn = feats_global + query 
        
        # 计算层级权重
        weights = self.scale_attention(fusion_for_attn) # [B, 4]
        weights_view = weights.view(b, n, 1, 1, 1)
        
        # 加权求和
        fused_feat = torch.sum(stacked_feats * weights_view, dim=1) # [B, C, H, W]
        
        out = self.project(fused_feat)
        return out, weights

# ==========================================
# 主模型 GazeLLE
# ==========================================
class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(512, 512), out_size=(64, 64),
                 use_sasa=False, use_ggsf=False, use_aux=False):
        """
        Args:
            use_sasa: 是否启用动态层级选择 (Contribution 1)
            use_ggsf: 是否启用几何门控 (Contribution 2)
            use_aux:  是否启用辅助监督 (Contribution 3)
        """
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.use_sasa = use_sasa
        self.use_ggsf = use_ggsf
        self.use_aux = use_aux
        
        # 打印当前配置，防止跑错
        print(f"Init GazeLLE with: SASA={use_sasa}, GGSF={use_ggsf}, AUX={use_aux}")

        # 1. 特征适配层: 将 Backbone 的 1024 维降到 256 维
        # 注意：这里我们对每一层都独立降维，方便后续处理
        self.feat_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone.get_dimension(), self.dim, 1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            ) for _ in range(4) # 假设 DinoV3Backbone 返回 4 层
        ])

        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # === 模块初始化 ===
        
        # [C1] SASA
        if self.use_sasa:
            self.sasa = ScaleAwareSemanticAggregator(self.dim, num_scales=4)
        else:
            # Fallback: 如果不用 SASA，就用简单的 Concat + Conv 融合 (还原你的 Baseline)
            # 输入通道是 dim * 4，输出 dim
            self.fallback_fusion = nn.Sequential(
                nn.Conv2d(self.dim * 4, self.dim, 1),
                nn.BatchNorm2d(self.dim),
                nn.ReLU()
            )

        # [C2] GGSF
        if self.use_ggsf:
            self.ggsf = GeometryGuidedSpatialFocus(self.featmap_h, self.featmap_w)

        # [C3] Aux Head
        if self.use_aux:
            self.aux_head = nn.Sequential(
                nn.ConvTranspose2d(self.dim, self.dim, kernel_size=2, stride=2),
                nn.Conv2d(self.dim, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        # Gazelle 原有组件
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        if self.inout: self.inout_token = nn.Embedding(1, self.dim)
        
        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for i in range(num_layers)
        ])
        
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.inout: 
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid()
            )

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. Backbone 提取多层特征
        raw_features_list = self.backbone.forward(input["images"])
        
        # 2. 特征适配 & Repeat
        adapted_features = []
        for i, feat in enumerate(raw_features_list):
            feat = self.feat_adapter[i](feat)
            feat = utils.repeat_tensors(feat, num_ppl_per_img) # [Total_People, 256, H, W]
            adapted_features.append(feat)

        # === [C2] GGSF 逻辑 ===
        if self.use_ggsf:
            geo_mask = self.ggsf(input["bboxes"], num_ppl_per_img, adapted_features[0].device)
            # 乘性门控
            gated_features = []
            for feat in adapted_features:
                gated_features.append(feat * geo_mask)
        else:
            # 如果不开启，直接透传
            gated_features = adapted_features

        # === [C3] Aux Head 逻辑 ===
        aux_preds = None
        if self.use_aux:
            # 使用最浅层 (Index 0) 进行监督
            shallow_feat = gated_features[0] 
            aux_out = self.aux_head(shallow_feat)
            aux_out = torchvision.transforms.functional.resize(aux_out, self.out_size).squeeze(1)
            aux_preds = utils.split_tensors(aux_out, num_ppl_per_img)

        # === [C1] SASA 逻辑 ===
        layer_weights = None
        if self.use_sasa:
            x, layer_weights = self.sasa(gated_features, self.head_token.weight)
        else:
            # Fallback: Concat -> Conv
            cat_feat = torch.cat(gated_features, dim=1) # [Total_People, 256*4, H, W]
            x = self.fallback_fusion(cat_feat)          # [Total_People, 256, H, W]

        # === 以下是 Gazelle 原有逻辑 ===
        x = x + self.pos_embed
        
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x + head_map_embeddings
        
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] 
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {
            "heatmap": heatmap_preds, 
            "aux_heatmap": aux_preds, 
            "inout": inout_preds if self.inout else None,
            "layer_weights": layer_weights 
        }

    def get_input_head_maps(self, bboxes):
        # 保持原有的辅助函数不变
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: 
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
        
def positionalencoding2d(d_model, height, width):
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

# ==========================================
# 工厂函数修改 (方便调用)
# ==========================================
def get_gazelle_model(model_name, use_sasa=False, use_ggsf=False, use_aux=False):
    # 工厂模式：根据名字选择对应的构造函数
    factory = {
        "gazelle_dinov3_vitb16": gazelle_dinov3_vitb16,
        "gazelle_dinov3_vitl16": gazelle_dinov3_vitl16,
        "gazelle_dinov3_vitb16_inout": gazelle_dinov3_vitb16_inout,
        "gazelle_dinov3_vitl16_inout": gazelle_dinov3_vitl16_inout,
    }
    assert model_name in factory.keys(), "invalid model name"
    
    # 将开关参数传递给具体的构造函数
    return factory[model_name](sasa=use_sasa, ggsf=use_ggsf, aux=use_aux)

def gazelle_dinov3_vitb16(sasa, ggsf, aux):
    backbone = DinoV3Backbone('dinov3_vitb16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux)
    return model, transform

def gazelle_dinov3_vitl16(sasa, ggsf, aux):
    backbone = DinoV3Backbone('dinov3_vitl16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux)
    return model, transform

def gazelle_dinov3_vitb16_inout(sasa, ggsf, aux):
    backbone = DinoV3Backbone('dinov3_vitb16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux)
    return model, transform

def gazelle_dinov3_vitl16_inout(sasa, ggsf, aux):
    backbone = DinoV3Backbone('dinov3_vitl16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux)
    return model, transform