import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import torch.nn.functional as F

import gazelle.utils as utils
from gazelle.ablation_variants import (
    CoordConvSpatialAdapter,
    EqualWeightFusion,
    FPNFusion,
    FixedGaussianSpatialPrior,
    FixedSectorSpatialPrior,
    FUSION_CHOICES,
    IdentitySpatialPrior,
    RawConcatFusion,
    SPATIAL_PRIOR_CHOICES,
    SelectedLayersFusion,
)
from gazelle.backbone import DinoV3Backbone

# ==========================================
# 模块 1: GGSF (几何引导) - 升级为尺度感知 (Scale-Aware)
# ==========================================
class GeometryGuidedSpatialFocus(nn.Module):
    def __init__(self, feat_h, feat_w, dropout=0.1):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        
        # 【关键升级】输入通道从 2 变为 4：相对坐标 (dx, dy) + Bbox宽高 (w, h)
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # 初始化 bias 偏大，保证初始阶段 mask 值接近 1，不破坏 DINO 原始特征，稳定前期训练
        nn.init.constant_(self.geo_mlp[-2].bias, 2.0) 

    def forward(self, bboxes, device):
        # 1. 构造基础网格 (0~1)
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
                    # 默认值
                    cx, cy, w, h = 0.5, 0.5, 1.0, 1.0
                else:
                    xmin, ymin, xmax, ymax = bbox
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin # 获取 Bbox 的物理尺寸
                
                # 2. 计算相对坐标 (dx, dy)
                center = torch.tensor([cx, cy], device=device).view(2, 1, 1)
                relative_coords = base_grid - center # [2, H, W]
                
                # 3. 将 w 和 h 扩展为特征图大小的通道
                w_tensor = torch.full((1, self.feat_h, self.feat_w), w, device=device) # [1, H, W]
                h_tensor = torch.full((1, self.feat_h, self.feat_w), h, device=device) # [1, H, W]
                
                # 4. 拼接成 4 通道几何特征
                # 这使得网络能同时知道“相对位置”和“头部大小”
                geo_feat = torch.cat([relative_coords, w_tensor, h_tensor], dim=0) # [4, H, W]
                final_geo_masks.append(geo_feat)
        
        if len(final_geo_masks) == 0:
            return None

        # [Total_People, 4, H, W]
        geo_input = torch.stack(final_geo_masks) 
        
        # [Total_People, 1, H, W] -> 0~1 的权重掩码
        attention_mask = self.geo_mlp(geo_input) 
        
        return attention_mask

# ==========================================
# 模块 2: SASA (动态融合) - 融合原始高维特征
# ==========================================
class ScaleAwareSemanticAggregator(nn.Module):
    def __init__(self, in_dim, num_scales=4, dropout=0.1):
        """
        in_dim: 输入特征的通道数 (Backbone原始维度，如 1024)
        """
        super().__init__()
        self.num_scales = num_scales
        
        # 1. 注意力网络
        # 我们需要给每个 Scale 算一个分数
        # 输入: [B, num_scales, in_dim]
        # 这里的 Linear 是作用在最后一维 (in_dim) 上的
        self.scale_attention = nn.Sequential(
            nn.Linear(in_dim, 128),      # [B, 4, 1024] -> [B, 4, 128]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),           # [B, 4, 128] -> [B, 4, 1] 【关键修复：输出变成1】
            # 我们不需要在这里 Softmax，因为我们要拿出 [B, 4] 之后再 Softmax
        )
        
        self.softmax = nn.Softmax(dim=1) # 在 Scale 维度 (dim=1) 上归一化

    def forward(self, features_list, head_token=None):
        # features_list: List of [B, C, H, W] (C=1024)
        
        # 1. 堆叠: [B, 4, C, H, W]
        stacked_feats = torch.stack(features_list, dim=1) 
        b, n, c, h, w = stacked_feats.shape
        
        # 2. 全局池化获取语义向量: [B, 4, C]
        feats_global = torch.mean(stacked_feats, dim=[3, 4]) 
        
        # 3. 计算权重
        # 注意：这里我们暂时忽略 head_token，避免维度不匹配问题 (1024 vs 256)
        # SASA 的核心是根据"图像内容的语义强度"来分配权重，光靠 feats_global 足够了
        
        # MLP: [B, 4, C] -> [B, 4, 1]
        attn_score = self.scale_attention(feats_global)
        
        # Squeeze & Softmax: [B, 4, 1] -> [B, 4] -> Softmax
        attn_score = attn_score.squeeze(-1) 
        weights = self.softmax(attn_score) # [B, 4]
        
        # 4. 准备加权: [B, 4] -> [B, 4, 1, 1, 1]
        weights_view = weights.view(b, n, 1, 1, 1)
        
        # 5. 加权融合: sum([B, 4, C, H, W] * [B, 4, 1, 1, 1]) -> [B, C, H, W]
        # fused_feat = torch.sum(stacked_feats * weights_view, dim=1)
        # 【关键修改】不要 sum，而是乘回去
        # [B, 4, C, H, W] * [B, 4, 1, 1, 1] -> [B, 4, C, H, W]
        weighted_feats_stack = stacked_feats * weights_view
        # 把它拆回 list，为了后面做 concat
        # 也可以直接在这里 reshape，但为了兼容性，我们返回处理后的 list
        weighted_features_list = [weighted_feats_stack[:, i, ...] for i in range(n)]
        
        return weighted_features_list, weights

# ==========================================
# 主模型 GazeLLE
# ==========================================
class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(512, 512), out_size=(64, 64),
                 use_sasa=False, use_ggsf=False, use_aux=False, dropout=0.1, spatial_prior=None,
                 fusion=None, selected_layers=None):
        super().__init__()
        self.backbone = backbone
        self.dim = dim # 最终 Transformer 的维度 (256)
        
        # 获取 Backbone 的原始输出维度 (e.g. 1024 for ViT-L)
        # 注意：这里调用的是 backbone.get_dimension()，但我们在 backbone.py 里改成了返回单层维度
        self.raw_dim = backbone.get_dimension() 
        
        self.spatial_prior = spatial_prior if spatial_prior is not None else ("ggsf" if use_ggsf else "none")
        if self.spatial_prior not in SPATIAL_PRIOR_CHOICES:
            raise ValueError(f"invalid spatial_prior: {self.spatial_prior}")

        self.fusion = fusion if fusion is not None else ("sasa" if use_sasa else "raw_concat")
        if self.fusion not in FUSION_CHOICES:
            raise ValueError(f"invalid fusion: {self.fusion}")

        self.use_sasa = self.fusion == "sasa"
        self.use_ggsf = self.spatial_prior == "ggsf"
        self.use_aux = use_aux
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        
        print(f"Init GazeLLE: Fusion={self.fusion}, SpatialPrior={self.spatial_prior}, AUX={use_aux}, RawDim={self.raw_dim}")

        self.linear = nn.Conv2d(self.raw_dim * 4, self.dim, 1)
        if self.use_sasa:
            self.sasa = ScaleAwareSemanticAggregator(self.raw_dim, num_scales=4, dropout=dropout)
        elif self.fusion == "fpn":
            self.fusion_module = FPNFusion(self.raw_dim, self.dim, num_layers=4)
        elif self.fusion == "selected_layers":
            self.fusion_module = SelectedLayersFusion(self.raw_dim, self.dim, selected_layers=selected_layers)

        self.spatial_prior_module = None
        self.coordconv_adapter = None
        if self.spatial_prior == "ggsf":
            self.ggsf = GeometryGuidedSpatialFocus(self.featmap_h, self.featmap_w, dropout=dropout)
        elif self.spatial_prior == "none":
            self.spatial_prior_module = IdentitySpatialPrior(self.featmap_h, self.featmap_w)
        elif self.spatial_prior == "fixed_gaussian":
            self.spatial_prior_module = FixedGaussianSpatialPrior(self.featmap_h, self.featmap_w)
        elif self.spatial_prior == "fixed_sector":
            self.spatial_prior_module = FixedSectorSpatialPrior(self.featmap_h, self.featmap_w)
        elif self.spatial_prior == "coordconv":
            self.coordconv_adapter = CoordConvSpatialAdapter(self.raw_dim, self.featmap_h, self.featmap_w)

        # Aux Head
        if self.use_aux:
            # 输入改为 self.raw_dim (例如 1024)
            self.aux_head = nn.Sequential(
                # 先用 3x3 卷积降维，提取浅层局部空间特征
                nn.Conv2d(self.raw_dim, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # 上采样到 Heatmap 尺寸
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.Conv2d(256, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        # 通用组件 (保持不变)
        self.num_layers = num_layers
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        if self.inout: self.inout_token = nn.Embedding(1, self.dim)
        self.transformer = nn.Sequential(*[Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1) for i in range(num_layers)])
        self.heatmap_head = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2), nn.Conv2d(dim, 1, kernel_size=1, bias=False), nn.Sigmoid())
        if self.inout: self.inout_head = nn.Sequential(nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. 获取原始多层特征 [List of (B, 1024, H, W)]
        raw_features_list = self.backbone.forward(input["images"])
        
        # 2. 将 Image 维度 repeat 成 Person 维度
        # [Total_Ppl, 1024, H, W]
        person_features_list = []
        for feat in raw_features_list:
            feat = utils.repeat_tensors(feat, num_ppl_per_img)
            person_features_list.append(feat)

        geo_mask = None
        if self.spatial_prior == "ggsf":
            geo_mask = self.ggsf(input["bboxes"], person_features_list[0].device)
            # 乘性门控
            gated_features = []
            for feat in person_features_list:
                gated_features.append(feat * geo_mask)
            processing_features = gated_features
        elif self.spatial_prior == "coordconv":
            processing_features = self.coordconv_adapter(person_features_list, input["bboxes"])
        else:
            geo_mask = self.spatial_prior_module(input["bboxes"], person_features_list[0].device)
            processing_features = [feat * geo_mask for feat in person_features_list]

        # === [C1] SASA vs Baseline ===
        if self.use_sasa:
            weighted_features_list, layer_weights = self.sasa(processing_features)
            cat_feat = torch.cat(weighted_features_list, dim=1)
            x = self.linear(cat_feat)
            fusion_metadata = {"fusion": "sasa", "uses_sasa_routing": True}
        elif self.fusion == "raw_concat":
            cat_feat = torch.cat(processing_features, dim=1)
            x = self.linear(cat_feat)
            layer_weights = None
            fusion_metadata = {"fusion": "raw_concat", "uses_sasa_routing": False, "num_layers": len(processing_features)}
        elif self.fusion == "equal_weight":
            layer_weight = 1.0 / len(processing_features)
            weighted_features = [feature * layer_weight for feature in processing_features]
            cat_feat = torch.cat(weighted_features, dim=1)
            x = self.linear(cat_feat)
            layer_weights = None
            fusion_metadata = {
                "fusion": "equal_weight",
                "uses_sasa_routing": False,
                "layer_weights": [layer_weight for _ in processing_features],
            }
        else:
            x, fusion_metadata = self.fusion_module(processing_features)
            layer_weights = None

        # === [C3] Aux Head (建议放在 linear 降维之后做) ===
        # 这样 Aux Head 参数少，且利用了融合后的特征
        aux_preds = None
        if self.use_aux:
            # 【关键修改】提取最浅层特征 (Layer 2 或 Layer 4)
            # 注意：这里的 processing_features 是已经经过 GGSF 门控的特征
            shallow_feat = processing_features[0] # [Total_People, 1024, H, W]
            
            # 预测粗糙热图
            aux_out = self.aux_head(shallow_feat) # [Total_People, 1, H*2, W*2]
            aux_out = torchvision.transforms.functional.resize(aux_out, self.out_size)
            
            # squeeze(1) 去掉通道维，得到 [Total_People, 64, 64]
            aux_preds = utils.split_tensors(aux_out.squeeze(1), num_ppl_per_img)

        # === 以下逻辑完全保持 Gazelle 原样 ===
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
            "layer_weights": layer_weights,
            "geo_mask": geo_mask,
            "spatial_prior": self.spatial_prior,
            "fusion": self.fusion,
            "fusion_metadata": fusion_metadata,
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
def get_gazelle_model(model_name, use_sasa=False, use_ggsf=False, use_aux=False, spatial_prior=None,
                      fusion=None, selected_layers=None):
    # 工厂模式：根据名字选择对应的构造函数
    factory = {
        "gazelle_dinov3_vitb16": gazelle_dinov3_vitb16,
        "gazelle_dinov3_vitl16": gazelle_dinov3_vitl16,
        "gazelle_dinov3_vitb16_inout": gazelle_dinov3_vitb16_inout,
        "gazelle_dinov3_vitl16_inout": gazelle_dinov3_vitl16_inout,
    }
    assert model_name in factory.keys(), "invalid model name"
    
    # 将开关参数传递给具体的构造函数
    return factory[model_name](
        sasa=use_sasa,
        ggsf=use_ggsf,
        aux=use_aux,
        spatial_prior=spatial_prior,
        fusion=fusion,
        selected_layers=selected_layers,
    )

def gazelle_dinov3_vitb16(sasa, ggsf, aux, spatial_prior=None, fusion=None, selected_layers=None):
    backbone = DinoV3Backbone('dinov3_vitb16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux, spatial_prior=spatial_prior, fusion=fusion, selected_layers=selected_layers)
    return model, transform

def gazelle_dinov3_vitl16(sasa, ggsf, aux, spatial_prior=None, fusion=None, selected_layers=None):
    backbone = DinoV3Backbone('dinov3_vitl16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux, spatial_prior=spatial_prior, fusion=fusion, selected_layers=selected_layers)
    return model, transform

def gazelle_dinov3_vitb16_inout(sasa, ggsf, aux, spatial_prior=None, fusion=None, selected_layers=None):
    backbone = DinoV3Backbone('dinov3_vitb16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux, spatial_prior=spatial_prior, fusion=fusion, selected_layers=selected_layers, inout=True)
    return model, transform

def gazelle_dinov3_vitl16_inout(sasa, ggsf, aux, spatial_prior=None, fusion=None, selected_layers=None):
    backbone = DinoV3Backbone('dinov3_vitl16')
    transform = backbone.get_transform((512, 512))
    model = GazeLLE(backbone, use_sasa=sasa, use_ggsf=ggsf, use_aux=aux, spatial_prior=spatial_prior, fusion=fusion, selected_layers=selected_layers, inout=True)
    return model, transform
