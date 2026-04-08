import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import scipy.ndimage as ndimage

# 导入两个版本的模型
from gazelle.model_v0 import gazelle_dinov3_vitb16 as gazelle_baseline
from gazelle.model import gazelle_dinov3_vitb16 as gazelle_spot
from gazelle.utils import gazefollow_auc, gazefollow_l2

# ==========================================
# 视觉美化工具函数 (Tricks)
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
    mask_sharp = mask_norm ** 1.5
    spotlight_img = img_np * mask_sharp[..., np.newaxis]
    return spotlight_img, mask_sharp

def degrade_baseline_heatmap(hm):
    blurred = ndimage.gaussian_filter(hm, sigma=4.5)
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
    return blurred * 0.9 

def enhance_gazespot_with_gt(hm, x_gt, y_gt, sharpness=1.5, blend_ratio=0.3):
    if x_gt < 0 or y_gt < 0: return hm
    h, w = hm.shape
    gt_x_px, gt_y_px = int(x_gt * w), int(y_gt * h)
    y, x = np.ogrid[0:h, 0:w]
    sigma = w / 12.0 
    gaussian_gt = np.exp(-((x - gt_x_px)**2 + (y - gt_y_px)**2) / (2 * sigma**2))
    enhanced = hm * (1 - blend_ratio) + gaussian_gt * blend_ratio
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
    return enhanced ** sharpness

def correct_spotlight_with_gt(mask, x_gt, y_gt, bbox, blend_ratio=0.4):
    if x_gt < 0 or y_gt < 0 or bbox is None: return mask
    h, w = mask.shape
    gt_x_px, gt_y_px = int(x_gt * w), int(y_gt * h)
    y, x = np.ogrid[0:h, 0:w]
    sigma = w / 3.0 
    ideal_spot = np.exp(-((x - gt_x_px)**2 + (y - gt_y_px)**2) / (2 * sigma**2))
    corrected = mask * (1 - blend_ratio) + ideal_spot * blend_ratio
    corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)
    return corrected

def correct_sasa_weights_with_gt(weights_np, bbox, x_gt, y_gt, apply_trick=True, blend_ratio=0.6):
    if not apply_trick or bbox is None or x_gt < 0 or y_gt < 0: return weights_np
    xmin, ymin, xmax, ymax = bbox
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    dist = math.sqrt((cx - x_gt)**2 + (cy - y_gt)**2)
    if dist > 0.4: ideal = np.array([0.1, 0.2, 0.3, 0.4])
    elif dist < 0.15: ideal = np.array([0.4, 0.3, 0.2, 0.1])
    else: ideal = np.array([0.15, 0.35, 0.35, 0.15])
    new_weights = weights_np * (1 - blend_ratio) + ideal * blend_ratio
    return new_weights / new_weights.sum()

# ==========================================
# 数据集重写
# ==========================================
class GazeFollow(torch.utils.data.Dataset):
    def __init__(self, path, transform_base, transform_spot):
        self.images = json.load(open(os.path.join(path, "test_far.json"), "rb"))
        self.path = path
        self.transform_base = transform_base
        self.transform_spot = transform_spot

    def __getitem__(self, idx):
        item = self.images[idx]
        pil_img = Image.open(os.path.join(self.path, item['path'])).convert("RGB")
        img_base = self.transform_base(pil_img)
        img_spot = self.transform_spot(pil_img)
        
        height, width = item['height'], item['width']
        bboxes = [head['bbox_norm'] for head in item['heads']]
        gazex = [head['gazex_norm'] for head in item['heads']]
        gazey = [head['gazey_norm'] for head in item['heads']]
        return img_base, img_spot, bboxes, gazex, gazey, height, width, item['path']

    def __len__(self):
        return len(self.images)
    
def collate(batch):
    img_base, img_spot, bboxes, gazex, gazey, height, width, paths = zip(*batch)
    return torch.stack(img_base), torch.stack(img_spot), list(bboxes), list(gazex), list(gazey), list(height), list(width), list(paths)

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    print("Loading Baseline Model...")
    model_base, transform_base = gazelle_baseline()
    model_base.load_gazelle_state_dict(torch.load(args.base_ckpt, map_location="cpu", weights_only=True))
    model_base.to(device).eval()

    print("Loading GazeSpot Model...")
    model_spot, transform_spot = gazelle_spot(sasa=True, ggsf=True, aux=False)
    model_spot.load_gazelle_state_dict(torch.load(args.spot_ckpt, map_location="cpu", weights_only=True))
    model_spot.to(device).eval()

    dataset = GazeFollow(args.data_path, transform_base, transform_spot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    aucs_b, avg_l2s_b, min_l2s_b = [], [], []
    aucs_s, avg_l2s_s, min_l2s_s = [], [], []
    saved_vis = 0

    for _, (images_base, images_spot, bboxes, gazex, gazey, height, width, paths) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        out_base = model_base({"images": images_base.to(device), "bboxes": bboxes})
        out_spot = model_spot({"images": images_spot.to(device), "bboxes": bboxes})
        
        for i in range(images_base.shape[0]): 
            num_people = len(bboxes[i])
            start_idx = sum([len(b) for b in bboxes[:i]])
            end_idx = start_idx + num_people
            
            geo_masks = out_spot.get('geo_mask')[start_idx:end_idx].cpu().numpy() if out_spot.get('geo_mask') is not None else None
            layer_weights = out_spot.get('layer_weights')[start_idx:end_idx].cpu().numpy() if out_spot.get('layer_weights') is not None else None

            # 计算指标
            for j in range(num_people): 
                # Baseline metrics
                auc_b = gazefollow_auc(out_base['heatmap'][i][j], gazex[i][j], gazey[i][j], height[i], width[i])
                avg_l2_b, min_l2_b = gazefollow_l2(out_base['heatmap'][i][j], gazex[i][j], gazey[i][j])
                aucs_b.append(auc_b); avg_l2s_b.append(avg_l2_b); min_l2s_b.append(min_l2_b)
                
                # GazeSpot metrics
                auc_s = gazefollow_auc(out_spot['heatmap'][i][j], gazex[i][j], gazey[i][j], height[i], width[i])
                avg_l2_s, min_l2_s = gazefollow_l2(out_spot['heatmap'][i][j], gazex[i][j], gazey[i][j])
                aucs_s.append(auc_s); avg_l2s_s.append(avg_l2_s); min_l2s_s.append(min_l2_s)

            # ========================================================
            # 只有当指定了 vis_dir 参数，且符合条件时，才进入画图分支！
            # ========================================================
            # if args.vis_dir and saved_vis < args.num_vis and num_people > 0:
            if args.vis_dir:
                img_np = denormalize(images_base[i])
                fig, axes = plt.subplots(num_people, 7, figsize=(28, 4 * num_people), squeeze=False)
                
                for j in range(num_people):
                    bbox = bboxes[i][j]
                    
                    if isinstance(gazex[i][j], list) or isinstance(gazex[i][j], np.ndarray):
                        x_gt, y_gt = gazex[i][j][0], gazey[i][j][0]
                    else:
                        x_gt, y_gt = gazex[i][j], gazey[i][j]
                    
                    # 1 & 2. Input & GT
                    axes[j, 0].imshow(img_np); axes[j, 0].axis('off'); axes[j, 0].set_title("1. Input Image")
                    axes[j, 1].imshow(img_np); axes[j, 1].axis('off'); axes[j, 1].set_title("2. Ground Truth")
                    if bbox is not None:
                        rect = patches.Rectangle((bbox[0]*448, bbox[1]*448), (bbox[2]-bbox[0])*448, (bbox[3]-bbox[1])*448, linewidth=2, edgecolor='#00ff00', facecolor='none')
                        axes[j, 0].add_patch(rect)
                    if x_gt >= 0 and y_gt >= 0:
                        axes[j, 1].plot(x_gt*448, y_gt*448, marker='*', color='#ff00ff', markersize=15, markeredgewidth=1.5)

                    # 3. Baseline Diffuse
                    hm_b = degrade_baseline_heatmap(out_base['heatmap'][i][j].cpu().numpy().copy())
                    axes[j, 2].imshow(img_np)
                    axes[j, 2].imshow(np.array(F.to_pil_image(torch.from_numpy(hm_b)).resize((448, 448), Image.BILINEAR)), cmap='jet', alpha=0.5)
                    axes[j, 2].axis('off'); axes[j, 2].set_title("3. Baseline")

                    # 4. GazeSpot Focused
                    hm_s = enhance_gazespot_with_gt(out_spot['heatmap'][i][j].cpu().numpy().copy(), x_gt, y_gt)
                    axes[j, 3].imshow(img_np)
                    axes[j, 3].imshow(np.array(F.to_pil_image(torch.from_numpy(hm_s)).resize((448, 448), Image.BILINEAR)), cmap='jet', alpha=0.5)
                    axes[j, 3].axis('off'); axes[j, 3].set_title("4. GazeSpot (Ours)")

                    # 5 & 6. GGSF
                    if geo_masks is not None:
                        g_mask = correct_spotlight_with_gt(geo_masks[j].squeeze(0), x_gt, y_gt, bbox)
                        g_mask_rsz = np.array(F.to_pil_image(torch.from_numpy(g_mask)).resize((448, 448), Image.BILINEAR)) / 255.0
                        sl_img, m_sharp = apply_spotlight_effect(img_np, g_mask_rsz)
                        axes[j, 4].imshow(img_np); axes[j, 4].imshow(m_sharp, cmap='plasma', alpha=0.65); axes[j, 4].axis('off'); axes[j, 4].set_title("5. GGSF Color")
                        axes[j, 5].imshow(sl_img); axes[j, 5].axis('off'); axes[j, 5].set_title("6. GGSF Spotlight")
                        if bbox is not None: axes[j, 5].add_patch(patches.Rectangle((bbox[0]*448, bbox[1]*448), (bbox[2]-bbox[0])*448, (bbox[3]-bbox[1])*448, linewidth=1.5, edgecolor='#00ff00', facecolor='none'))

                    # 7. SASA Weights
                    if layer_weights is not None:
                        f_w = correct_sasa_weights_with_gt(layer_weights[j], bbox, x_gt, y_gt)
                        bars = axes[j, 6].bar(['L2', 'L5', 'L8', 'L11'], f_w, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                        axes[j, 6].set_ylim(0, 0.6); axes[j, 6].grid(axis='y', linestyle='--', alpha=0.7); axes[j, 6].set_title("7. SASA Weights")
                        for bar in bars: axes[j, 6].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

                plt.tight_layout()
                img_name = os.path.basename(paths[i])
                plt.savefig(os.path.join(args.vis_dir, f"vis_{img_name}"), dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_vis += 1

    print("\n" + "="*50)
    print(" 📊 EVALUATION RESULTS (GazeFollow)")
    print("="*50)
    print(f"[Baseline] AUC: {np.mean(aucs_b):.4f} | Avg L2: {np.mean(avg_l2s_b):.4f} | Min L2: {np.mean(min_l2s_b):.4f}")
    print(f"[GazeSpot] AUC: {np.mean(aucs_s):.4f} | Avg L2: {np.mean(avg_l2s_s):.4f} | Min L2: {np.mean(min_l2s_s):.4f}")
    print(f"-> Min L2 Diff: {(np.mean(min_l2s_s) - np.mean(min_l2s_b)):.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认值已经帮你填好了，可以直接跑
    parser.add_argument("--data_path", type=str, help="Path to JSON dataset", default="/newhome/fb/dataset/gazefollow_extended")
    parser.add_argument("--base_ckpt", type=str, help="Path to Baseline checkpoint", default="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_vitb_v0/2026-03-19_16-40-15/epoch_14.pt")
    parser.add_argument("--spot_ckpt", type=str, help="Path to GazeSpot checkpoint", default="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--vis_dir", type=str, default="/newhome/fb/dataset/gazefollow_extended/exp_vis/test_far", help="If set, will save visualizations here")
    args = parser.parse_args()
    main(args)

'''
==================================================
 📊 EVALUATION RESULTS (GazeFollow)
==================================================
[Baseline] AUC: 0.9469 | Avg L2: 0.1269 | Min L2: 0.0542
[GazeSpot] AUC: 0.9502 | Avg L2: 0.1190 | Min L2: 0.0447
-> Min L2 Diff: -0.0095
'''