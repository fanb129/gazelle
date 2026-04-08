import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

# ==========================================
# 导入模型工厂 (请确保路径和你的代码一致)
# ==========================================
# 1. GazeFollow 模型 (无 inout 预测头)
# 注意：这里按你之前的上下文使用了 DINOv3，如果你想用 DINOv2，请自行改回 model_v0_dinov2
from gazelle.model_v0 import gazelle_dinov3_vitb16 as gazelle_baseline_gf
from gazelle.model import gazelle_dinov3_vitb16 as gazelle_spot_gf

# 2. VAT 模型 (带 inout 预测头)
from gazelle.model_v0 import gazelle_dinov3_vitb16_inout as gazelle_baseline_vat
from gazelle.model import gazelle_dinov3_vitb16_inout as gazelle_spot_vat

# ==========================================
# 模型包装器 (Wrapper)
# ==========================================
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dummy_bboxes = [[[0.3, 0.3, 0.6, 0.6]]]

    def forward(self, images):
        model_input = {
            "images": images,
            "bboxes": self.dummy_bboxes
        }
        return self.model(model_input)

def compute_model_complexity(model, model_name, input_size, device="cpu"):
    """
    计算并打印模型的 FLOPs 和 Params
    """
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()

    dummy_image = torch.randn(input_size).to(device)

    # 1. 使用 thop 计算总 FLOPs 和 总参数
    with torch.no_grad():
        macs, total_params = profile(wrapped_model, inputs=(dummy_image,), verbose=False)

    # 2. 手动计算可训练参数 (也就是你加的轻量级头部模块)
    # 因为 Backbone 冻结了，所以只需要统计不属于 backbone 的参数即可
    head_params = sum(p.numel() for name, p in model.named_parameters() if "backbone" not in name)
    
    macs_str, _ = clever_format([macs, total_params], "%.2f")

    print(f"[{model_name}] (Resolution: {input_size[2]}x{input_size[3]})")
    print(f" -> Total Params:     {total_params / 1e6:.2f} M (包含冻结的主干)")
    print(f" -> Head Params:      {head_params / 1e6:.2f} M (🟢 论文表格填这个: 可训练参数)")
    print(f" -> MACs (FLOPs):     {macs_str}        (🟢 论文表格填这个: 整体计算量)")
    print("-" * 60)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...\n")
    print("=" * 60)
    print(" 📊 PART 1: GazeFollow Dataset (No Inout Head)")
    print("=" * 60)

    # Baseline on GazeFollow (448x448)
    model_base_gf, _ = gazelle_baseline_gf()
    compute_model_complexity(model_base_gf, "Baseline (GazeFollow)", input_size=(1, 3, 448, 448), device=device)

    # GazeSpot on GazeFollow (512x512)
    model_spot_gf, _ = gazelle_spot_gf(sasa=True, ggsf=True, aux=False)
    compute_model_complexity(model_spot_gf, "GazeSpot (GazeFollow)", input_size=(1, 3, 512, 512), device=device)


    print("\n" + "=" * 60)
    print(" 📊 PART 2: VAT Dataset (With Inout Head)")
    print("=" * 60)

    # Baseline on VAT (448x448)
    model_base_vat, _ = gazelle_baseline_vat()
    compute_model_complexity(model_base_vat, "Baseline (VAT)", input_size=(1, 3, 448, 448), device=device)

    # GazeSpot on VAT (512x512)
    model_spot_vat, _ = gazelle_spot_vat(sasa=True, ggsf=True, aux=False)
    compute_model_complexity(model_spot_vat, "GazeSpot (VAT)", input_size=(1, 3, 512, 512), device=device)