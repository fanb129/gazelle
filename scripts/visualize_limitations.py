import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from PIL import Image

# 导入你的模型工厂
from gazelle.model import get_gazelle_model

# ==========================================
# 工具函数 (绝对真实，没有任何 GT 修正 Trick)
# ==========================================
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()

def apply_spotlight_effect(img_np, mask_np):
    mask_min, mask_max = mask_np.min(), mask_np.max()
    if mask_max - mask_min > 1e-5:
        mask_norm = (mask_np - mask_min) / (mask_max - mask_min)
    else:
        mask_norm = mask_np
        
    # 【核心修改】：将 1.5 提高到 3.0。
    # 指数越大，非最高点的值衰减越快，白色光晕越小，黑色背景区域就越大、越纯粹。
    mask_sharp = mask_norm ** 1.0 
    
    spotlight_img = img_np * mask_sharp[..., np.newaxis]
    return spotlight_img, mask_sharp

# ==========================================
# 主函数
# ==========================================
@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GazeSpot Model for Failure Case Analysis...")
    # 启用 SASA 和 GGSF，展示真实输出
    model, transform = get_gazelle_model(args.model_name, use_sasa=True, use_ggsf=True, use_aux=False)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True))
    model.to(device).eval()

    # 1. 读取 JSON 标注文件
    print(f"Loading annotations from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data_list = json.load(f)
        
    # 为了方便查找，构建一个以 json 内部 path 为 key 的字典
    # 例如 GazeFollow 的 path 通常是 "test2/00000004/00004001.jpg"
    data_dict = {item['path']: item for item in data_list}

    # 2. 读取用户提供的 .txt 路径列表
    print(f"Loading target image list from {args.img_list}...")
    with open(args.img_list, 'r') as f:
        # 去除换行符和空行
        target_img_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(target_img_paths)} images to process.")

    # 3. 开始精准处理
    for img_path in target_img_paths:
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image not found at {img_path}, skipping...")
            continue
            
        # 在 JSON 中寻找对应的标注信息
        matched_item = None
        for json_path_key in data_dict.keys():
            if img_path.endswith(json_path_key):
                matched_item = data_dict[json_path_key]
                break
                
        if matched_item is None:
            print(f"⚠️ Warning: No annotation found for {img_path} in JSON, skipping...")
            continue

        print(f"Processing Failure Case: {img_path}...")
        
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        bboxes = [head['bbox_norm'] for head in matched_item['heads']]
        
        # 兼容处理 GT 坐标
        gazex_norms, gazey_norms = [], []
        for head in matched_item['heads']:
            gx = head['gazex_norm'][0] if isinstance(head['gazex_norm'], list) else head['gazex_norm']
            gy = head['gazey_norm'][0] if isinstance(head['gazey_norm'], list) else head['gazey_norm']
            gazex_norms.append(gx)
            gazey_norms.append(gy)
        
        model_input = {"images": img_tensor, "bboxes": [bboxes]}
        # bad_bbox = [0.8, 0.8, 0.9, 0.9] 
        # model_input = {"images": img_tensor, "bboxes": [[bad_bbox]]}
        # 获取最原始的、发生错误的预测结果
        out_spot = model(model_input)
        
        heatmaps_spot = out_spot['heatmap'][0].cpu().numpy()
        geo_masks = out_spot.get('geo_mask').cpu() if out_spot.get('geo_mask') is not None else None
        img_np = denormalize(img_tensor[0])
        img_h, img_w = img_np.shape[:2] # 动态获取当前图片的宽高 (512或448)
        
        num_people = len(bboxes)
        fig, axes = plt.subplots(num_people, 4, figsize=(16, 4 * num_people), squeeze=False)
        
        for i in range(num_people):
            bbox = bboxes[i]
            x_gt, y_gt = gazex_norms[i], gazey_norms[i]
            
            # --- 列 1: Input + Bbox ---
            ax = axes[i, 0]
            ax.imshow(img_np)
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                rect = patches.Rectangle((xmin * img_w, ymin * img_h), (xmax - xmin) * img_w, (ymax - ymin) * img_h, 
                                         linewidth=2, edgecolor='#00ff00', facecolor='none')
                ax.add_patch(rect)
            ax.set_title("1. Input & Head Bbox", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # --- 列 2: Ground Truth ---
            ax = axes[i, 1]
            ax.imshow(img_np)
            if x_gt >= 0 and y_gt >= 0:
                ax.plot(x_gt * img_w, y_gt * img_h, marker='*', color='#ff00ff', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
            ax.set_title("2. Ground Truth", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # --- 列 3: Failed Spotlight (真实的 Geo Mask) ---
            ax = axes[i, 2]
            if geo_masks is not None:
                g_mask_raw = geo_masks[i].squeeze(0).numpy()
                g_mask_rsz = np.array(F.to_pil_image(torch.from_numpy(g_mask_raw)).resize((img_w, img_h), Image.BILINEAR)) / 255.0
                
                spotlight_img, mask_sharp = apply_spotlight_effect(img_np, g_mask_rsz)
                ax.imshow(spotlight_img)
                if bbox is not None:
                    rect = patches.Rectangle((xmin * img_w, ymin * img_h), (xmax - xmin) * img_w, (ymax - ymin) * img_h, 
                                             linewidth=1.5, edgecolor='#00ff00', facecolor='none')
                    ax.add_patch(rect)
                ax.set_title("3. Flawed Spatial Prior", color='red', fontsize=12, fontweight='bold')
            ax.axis('off')

            # --- 列 4: Failed Heatmap ---
            ax = axes[i, 3]
            hm_s = heatmaps_spot[i]
            hm_h, hm_w = hm_s.shape # 动态获取特征图大小
            hm_s_img = F.to_pil_image(torch.from_numpy(hm_s)).resize((img_w, img_h), Image.BILINEAR)
            ax.imshow(img_np)
            ax.imshow(np.array(hm_s_img), cmap='jet', alpha=0.6)
            
            # 标出错误的最高点
            y_max, x_max = np.unravel_index(hm_s.argmax(), hm_s.shape)
            ax.plot(x_max / hm_w * img_w, y_max / hm_h * img_h)
            
            ax.set_title("4. Incorrect Prediction", color='red', fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        # 提取原图的文件名作为保存的前缀
        img_basename = os.path.basename(img_path)
        save_path = os.path.join(args.output_dir, f"limitation_{img_basename}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) 
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_list", type=str, default="/home/fb/src/paper/gazelleV1/scripts/limit_gazefollow.txt", help="存放目标图片路径的 txt 文件")
    parser.add_argument("--output_dir", type=str, default="/home/fb/src/paper/gazelleV1/limitation_output/limit_gazefollow/", help="输出结果文件夹")
    parser.add_argument("--json_path", type=str, default="/newhome/fb/dataset/gazefollow_extended/test_preprocessed.json", help="对应的 JSON 标注文件")
    parser.add_argument("--ckpt_path", type=str, default="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt", help="GazeSpot 权重路径")
    parser.add_argument("--model_name", type=str, default="gazelle_dinov3_vitb16", help="VAT数据请加上 _inout")
    
    args = parser.parse_args()
    main(args)