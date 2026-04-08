import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

# 导入你的模型工厂
from gazelle.model_dinov2 import get_gazelle_model

# ==========================================
# 1. 动态 Hook 捕获器
# ==========================================
class FeatureCatcher:
    def __init__(self, model):
        self.features = {}
        self.hooks = []
        
        def hook_linear(module, inp, outp):
            # outp 形状 [B, 256, H, W]
            self.features['sasa_fused'] = outp.detach().cpu()

        self.hooks.append(model.linear.register_forward_hook(hook_linear))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# ==========================================
# 2. PCA 语义图处理核心
# ==========================================
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()

def draw_pca_semantic_map(ax, feat_tensor, title):
    """
    将高维特征图 [C, H, W] 经过 PCA 降维到 3 维 (RGB)，生成语义图
    """
    C, H, W = feat_tensor.shape
    
    # 1. 展平特征空间: [C, H*W] -> [H*W, C]
    feat_flat = feat_tensor.reshape(C, -1).numpy().T 
    
    # 2. PCA 降维提取前 3 个主成分
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(feat_flat) # [H*W, 3]
    
    # 3. Min-Max 归一化到 [0, 1] 以便作为 RGB 图像显示
    # 对每个通道独立归一化，确保色彩鲜艳
    pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0) + 1e-8)
    
    # 4. 还原为图像尺寸: [H, W, 3]
    pca_rgb = pca_features.reshape(H, W, 3)
    
    # 5. 为了平滑显示，使用 PIL 将 32x32 放大回高分辨率 (利用双三次插值)
    # 因为 PCA 出图直接是 RGB，我们不再用伪色彩 (cmap)
    pca_img = Image.fromarray((pca_rgb * 255).astype(np.uint8)).resize((448, 448), Image.BICUBIC)
    
    ax.imshow(pca_img)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

# ==========================================
# 3. 主函数
# ==========================================
@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GazeSpot Model...")
    model, transform = get_gazelle_model(args.model_name, use_sasa=True, use_ggsf=True, use_aux=False)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True))
    model.to(device).eval()

    catcher = FeatureCatcher(model)

    print(f"Processing image: {args.img_path}")
    pil_img = Image.open(args.img_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 传入真实的 Bbox
    real_bboxes = [[[args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]]]] 
    
    model_input = {
        "images": img_tensor,
        "bboxes": real_bboxes
    }

    # 1. 手动获取 DINOv3 输出
    raw_layers_gpu = model.backbone.forward(img_tensor)
    raw_layers = [feat.detach().cpu() for feat in raw_layers_gpu]
    
    # 2. 正常推断，获取融合特征
    _ = model(model_input)
    sasa_fused = catcher.features['sasa_fused'] 

    img_np = denormalize(img_tensor[0])

    # ==========================================
    # 4. 创建画板并绘制 PCA 语义图
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # 图 1: 原图 (把真实的框也画上去，证明输入无误)
    axes[0].imshow(img_np)
    axes[0].axis('off')
    axes[0].set_title("Original Image (with Head Bbox)", fontsize=14, fontweight='bold', pad=10)
    import matplotlib.patches as patches
    rect = patches.Rectangle((args.bbox[0]*448, args.bbox[1]*448), 
                             (args.bbox[2]-args.bbox[0])*448, 
                             (args.bbox[3]-args.bbox[1])*448, 
                             linewidth=2, edgecolor='#00ff00', facecolor='none')
    axes[0].add_patch(rect)

    # 图 2-5: DINOv3 原始 4 层特征的 PCA 语义图
    layer_names = ["Layer 2 PCA (Shallow Geo)", 
                   "Layer 5 PCA (Mid-level)", 
                   "Layer 8 PCA (Mid-level)", 
                   "Layer 11 PCA (Deep Semantic)"]
                   
    for i in range(4):
        draw_pca_semantic_map(axes[i+1], raw_layers[i][0], layer_names[i])

    # 图 6: SASA 融合后的特征 PCA
    draw_pca_semantic_map(axes[5], sasa_fused[0], "SASA Fused PCA\n(With GGSF Suppression)")

    plt.tight_layout()
    img_basename = os.path.basename(args.img_path)
    save_path = os.path.join(args.output_dir, f"pca_evolution_dinov2_{img_basename}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    catcher.remove_hooks()
    print(f"✅ PCA Semantic visualization saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="/newhome/fb/dataset/gazefollow_extended/test2/00000000/00000042.jpg", help="你想可视化的输入图片路径")
    parser.add_argument("--ckpt_path", type=str, default="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_dinov2_sasa_ggsf/2026-03-26_00-58-25/epoch_14.pt", help="GazeSpot checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="/home/fb/src/paper/gazelleV1/visualizations/feature_maps", help="生成的特征图保存路径")
    parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitb14", help="使用的模型名字")

    parser.add_argument("--bbox", nargs=4, type=float, default=[0, 0, 0.1, 0.1], help="真实的人头框: xmin ymin xmax ymax")

    parser.add_argument("--json_path", type=str)
    args = parser.parse_args()
    main(args)


'''
python scripts/visualize_features.py \
    --input_image "/home/fb/src/paper/gazelleV1/visualizations/NearSubset/compare_00001234.jpg" \
    --ckpt_path "./checkpoints/gazespot_epoch14.pt" \
    --output_dir "./visualizations/feature_maps"
'''