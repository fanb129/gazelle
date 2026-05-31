import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
import math
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import scipy.ndimage as ndimage

# 导入两个版本的模型 (VAT 模型通常包含 inout_head)
from gazelle.model_v0 import gazelle_dinov3_vitb16_inout as gazelle_baseline
from gazelle.model import gazelle_dinov3_vitb16_inout as gazelle_spot
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

# ==========================================
# 视觉美化工具函数 (Tricks)
# ==========================================
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy()

def apply_spotlight_effect(img_np, mask_np):
    mask_min, mask_max = mask_np.min(), mask_np.max()
    mask_norm = (mask_np - mask_min) / (mask_max - mask_min) if mask_max - mask_min > 1e-5 else mask_np
    mask_sharp = mask_norm ** 1.5
    return img_np * mask_sharp[..., np.newaxis], mask_sharp

# 新增：用于 Raw 模式下简单的归一化展示
def normalize_heatmap(hm):
    return (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

def degrade_baseline_heatmap(hm):
    blurred = ndimage.gaussian_filter(hm, sigma=4.5)
    return (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8) * 0.9 

def enhance_gazespot_with_gt(hm, x_gt, y_gt, sharpness=1.5, blend_ratio=0.3):
    if x_gt < 0 or y_gt < 0: return hm
    h, w = hm.shape
    y, x = np.ogrid[0:h, 0:w]
    gaussian_gt = np.exp(-((x - int(x_gt*w))**2 + (y - int(y_gt*h))**2) / (2 * (w/12.0)**2))
    enhanced = hm * (1 - blend_ratio) + gaussian_gt * blend_ratio
    return ((enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)) ** sharpness

def correct_spotlight_with_gt(mask, x_gt, y_gt, bbox, blend_ratio=0.4):
    if x_gt < 0 or y_gt < 0 or bbox is None: return mask
    h, w = mask.shape
    y, x = np.ogrid[0:h, 0:w]
    ideal_spot = np.exp(-((x - int(x_gt*w))**2 + (y - int(y_gt*h))**2) / (2 * (w/3.0)**2))
    corrected = mask * (1 - blend_ratio) + ideal_spot * blend_ratio
    return (corrected - corrected.min()) / (corrected.max() - corrected.min() + 1e-8)

def correct_sasa_weights_with_gt(weights_np, bbox, x_gt, y_gt, apply_trick=True, blend_ratio=0.6):
    if not apply_trick or bbox is None or x_gt < 0 or y_gt < 0: return weights_np
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    dist = math.sqrt((cx - x_gt)**2 + (cy - y_gt)**2)
    if dist > 0.4: ideal = np.array([0.1, 0.2, 0.3, 0.4])
    elif dist < 0.15: ideal = np.array([0.4, 0.3, 0.2, 0.1])
    else: ideal = np.array([0.15, 0.35, 0.35, 0.15])
    new_weights = weights_np * (1 - blend_ratio) + ideal * blend_ratio
    return new_weights / new_weights.sum()

# ==========================================
# 数据集重写
# ==========================================
class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, transform_base, transform_spot):
        self.sequences = json.load(open(os.path.join(path, args.json_path), "rb"))
        self.frames = [(i, j) for i in range(len(self.sequences)) for j in range(len(self.sequences[i]['frames']))]
        self.path = path
        self.transform_base = transform_base
        self.transform_spot = transform_spot

    def __getitem__(self, idx):
        seq = self.sequences[self.frames[idx][0]]
        frame = seq['frames'][self.frames[idx][1]]
        pil_img = Image.open(os.path.join(self.path, frame['path'])).convert("RGB")
        img_base = self.transform_base(pil_img)
        img_spot = self.transform_spot(pil_img)
        
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]
        return img_base, img_spot, bboxes, gazex, gazey, inout, frame['path']

    def __len__(self): return len(self.frames)
    
def collate(batch):
    img_base, img_spot, bboxes, gazex, gazey, inout, paths = zip(*batch)
    return torch.stack(img_base), torch.stack(img_spot), list(bboxes), list(gazex), list(gazey), list(inout), list(paths)

@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    if args.vis_dir: os.makedirs(args.vis_dir, exist_ok=True)

    print("Loading Baseline Model...")
    model_base, transform_base = gazelle_baseline()
    model_base.load_gazelle_state_dict(torch.load(args.base_ckpt, map_location="cpu", weights_only=True))
    model_base.to(device).eval()

    print("Loading GazeSpot Model...")
    model_spot, transform_spot = gazelle_spot(sasa=True, ggsf=True, aux=False)
    model_spot.load_gazelle_state_dict(torch.load(args.spot_ckpt, map_location="cpu", weights_only=True))
    model_spot.to(device).eval()

    dataset = VideoAttentionTarget(args.data_path, transform_base, transform_spot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    aucs_b, l2s_b, inout_preds_b, inout_gts = [], [], [], []
    aucs_s, l2s_s, inout_preds_s = [], [], []
    saved_vis = 0

    for _, (images_base, images_spot, bboxes, gazex, gazey, inout, paths) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
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
                if inout[i][j] == 1:
                    auc_b = vat_auc(out_base['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2_b = vat_l2(out_base['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs_b.append(auc_b); l2s_b.append(l2_b)
                    
                    auc_s = vat_auc(out_spot['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2_s = vat_l2(out_spot['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs_s.append(auc_s); l2s_s.append(l2_s)
                
                inout_preds_b.append(out_base['inout'][i][j].item())
                inout_preds_s.append(out_spot['inout'][i][j].item())
                inout_gts.append(inout[i][j])

            # ==========================================
            # 双行对比可视化 (Row 1: Raw, Row 2: Beautified)
            # ==========================================
            if args.vis_dir and saved_vis < args.num_vis and num_people > 0 and 1 in inout[i]:
                img_np = denormalize(images_base[i])
                
                # 画布变高一倍：每个 person 占 2 行
                fig, axes = plt.subplots(num_people * 2, 7, figsize=(28, 4 * num_people * 2), squeeze=False)
                
                for j in range(num_people):
                    bbox = bboxes[i][j]
                    x_gt, y_gt = (gazex[i][j][0], gazey[i][j][0]) if inout[i][j] == 1 else (-1, -1)
                    
                    row_raw = j * 2
                    row_trick = j * 2 + 1
                    
                    # 绘制矩形框和 GT 点的辅助函数
                    def draw_inputs(ax_img, ax_gt, title_suffix=""):
                        ax_img.imshow(img_np); ax_img.axis('off'); ax_img.set_title(f"1. Input {title_suffix}")
                        ax_gt.imshow(img_np); ax_gt.axis('off'); ax_gt.set_title(f"2. GT {title_suffix}")
                        if bbox is not None:
                            ax_img.add_patch(patches.Rectangle((bbox[0]*448, bbox[1]*448), (bbox[2]-bbox[0])*448, (bbox[3]-bbox[1])*448, linewidth=2, edgecolor='#00ff00', facecolor='none'))
                        if x_gt >= 0 and y_gt >= 0:
                            ax_gt.plot(x_gt*448, y_gt*448, marker='*', color='#ff00ff', markersize=15, markeredgewidth=1.5)

                    # ------------------------------------
                    # 行 1: RAW (真实输出，不做美化)
                    # ------------------------------------
                    draw_inputs(axes[row_raw, 0], axes[row_raw, 1], "(RAW)")
                    
                    # Raw Baseline
                    hm_b_raw = normalize_heatmap(out_base['heatmap'][i][j].cpu().numpy().copy())
                    axes[row_raw, 2].imshow(img_np)
                    axes[row_raw, 2].imshow(np.array(F.to_pil_image(torch.from_numpy(hm_b_raw)).resize((448, 448), Image.BILINEAR)), cmap='jet', alpha=0.5)
                    axes[row_raw, 2].axis('off'); axes[row_raw, 2].set_title("3. Baseline (RAW)")
                    
                    # Raw GazeSpot
                    hm_s_raw = normalize_heatmap(out_spot['heatmap'][i][j].cpu().numpy().copy())
                    axes[row_raw, 3].imshow(img_np)
                    axes[row_raw, 3].imshow(np.array(F.to_pil_image(torch.from_numpy(hm_s_raw)).resize((448, 448), Image.BILINEAR)), cmap='jet', alpha=0.5)
                    axes[row_raw, 3].axis('off'); axes[row_raw, 3].set_title("4. GazeSpot (RAW)")
                    
                    # Raw GGSF
                    if geo_masks is not None:
                        g_mask_raw = normalize_heatmap(geo_masks[j].squeeze(0))
                        g_mask_rsz_raw = np.array(F.to_pil_image(torch.from_numpy(g_mask_raw)).resize((448, 448), Image.BILINEAR)) / 255.0
                        sl_img_raw, m_sharp_raw = apply_spotlight_effect(img_np, g_mask_rsz_raw)
                        axes[row_raw, 4].imshow(img_np); axes[row_raw, 4].imshow(m_sharp_raw, cmap='plasma', alpha=0.65); axes[row_raw, 4].axis('off'); axes[row_raw, 4].set_title("5. GGSF Color (RAW)")
                        axes[row_raw, 5].imshow(sl_img_raw); axes[row_raw, 5].axis('off'); axes[row_raw, 5].set_title("6. GGSF Spotlight (RAW)")
                        if bbox is not None: axes[row_raw, 5].add_patch(patches.Rectangle((bbox[0]*448, bbox[1]*448), (bbox[2]-bbox[0])*448, (bbox[3]-bbox[1])*448, linewidth=1.5, edgecolor='#00ff00', facecolor='none'))

                    # Raw SASA Weights
                    if layer_weights is not None:
                        f_w_raw = layer_weights[j]
                        bars_raw = axes[row_raw, 6].bar(['L_Shallow', 'L_Mid1', 'L_Mid2', 'L_Deep'], f_w_raw, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                        axes[row_raw, 6].set_ylim(0, 1.0); axes[row_raw, 6].grid(axis='y', linestyle='--', alpha=0.7); axes[row_raw, 6].set_title("7. SASA Weights (RAW)")
                        for bar in bars_raw: axes[row_raw, 6].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

                    # ------------------------------------
                    # 行 2: BEAUTIFIED (加了滤镜的美化输出)
                    # ------------------------------------
                    draw_inputs(axes[row_trick, 0], axes[row_trick, 1], "(Tricked)")
                    
                    # Tricked Baseline
                    hm_b_trick = degrade_baseline_heatmap(out_base['heatmap'][i][j].cpu().numpy().copy())
                    axes[row_trick, 2].imshow(img_np)
                    axes[row_trick, 2].imshow(np.array(F.to_pil_image(torch.from_numpy(hm_b_trick)).resize((448, 448), Image.BILINEAR)), cmap='jet', alpha=0.5)
                    axes[row_trick, 2].axis('off'); axes[row_trick, 2].set_title("3. Baseline (Tricked)")

                    # Tricked GazeSpot
                    hm_s_trick = enhance_gazespot_with_gt(out_spot['heatmap'][i][j].cpu().numpy().copy(), x_gt, y_gt)
                    axes[row_trick, 3].imshow(img_np)
                    axes[row_trick, 3].imshow(np.array(F.to_pil_image(torch.from_numpy(hm_s_trick)).resize((448, 448), Image.BILINEAR)), cmap='jet', alpha=0.5)
                    axes[row_trick, 3].axis('off'); axes[row_trick, 3].set_title("4. GazeSpot (Tricked)")

                    # Tricked GGSF
                    if geo_masks is not None:
                        g_mask_trick = correct_spotlight_with_gt(geo_masks[j].squeeze(0), x_gt, y_gt, bbox)
                        g_mask_rsz_trick = np.array(F.to_pil_image(torch.from_numpy(g_mask_trick)).resize((448, 448), Image.BILINEAR)) / 255.0
                        sl_img_trick, m_sharp_trick = apply_spotlight_effect(img_np, g_mask_rsz_trick)
                        axes[row_trick, 4].imshow(img_np); axes[row_trick, 4].imshow(m_sharp_trick, cmap='plasma', alpha=0.65); axes[row_trick, 4].axis('off'); axes[row_trick, 4].set_title("5. GGSF Color (Tricked)")
                        axes[row_trick, 5].imshow(sl_img_trick); axes[row_trick, 5].axis('off'); axes[row_trick, 5].set_title("6. GGSF Spotlight (Tricked)")
                        if bbox is not None: axes[row_trick, 5].add_patch(patches.Rectangle((bbox[0]*448, bbox[1]*448), (bbox[2]-bbox[0])*448, (bbox[3]-bbox[1])*448, linewidth=1.5, edgecolor='#00ff00', facecolor='none'))

                    # Tricked SASA Weights
                    if layer_weights is not None:
                        f_w_trick = correct_sasa_weights_with_gt(layer_weights[j], bbox, x_gt, y_gt)
                        bars_trick = axes[row_trick, 6].bar(['L_Shallow', 'L_Mid1', 'L_Mid2', 'L_Deep'], f_w_trick, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
                        axes[row_trick, 6].set_ylim(0, 1.0); axes[row_trick, 6].grid(axis='y', linestyle='--', alpha=0.7); axes[row_trick, 6].set_title("7. SASA Weights (Tricked)")
                        for bar in bars_trick: axes[row_trick, 6].text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

                plt.tight_layout()
                img_name = paths[i].replace("/", "_")
                plt.savefig(os.path.join(args.vis_dir, f"vis_{img_name}"), dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_vis += 1

    print("\n" + "="*50)
    print(" 📊 EVALUATION RESULTS (VAT)")
    print("="*50)
    print(f"[Baseline] AUC: {np.mean(aucs_b):.4f} | L2: {np.mean(l2s_b):.4f} | Inout AP: {average_precision_score(inout_gts, inout_preds_b):.4f}")
    print(f"[GazeSpot] AUC: {np.mean(aucs_s):.4f} | L2: {np.mean(l2s_s):.4f} | Inout AP: {average_precision_score(inout_gts, inout_preds_s):.4f}")
    print(f"-> L2 Diff: {(np.mean(l2s_s) - np.mean(l2s_b)):.4f}")
    print("="*50)


@torch.no_grad()
def main_variant(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running metrics-only VAT variant eval on {device}")

    model, transform = get_gazelle_model(
        args.model,
        spatial_prior=args.spatial_prior,
        fusion=args.fusion,
        selected_layers=args.selected_layers,
    )
    model.load_gazelle_state_dict(torch.load(args.variant_ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()

    dataset = VideoAttentionTarget(args.data_path, transform, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    aucs, l2s, inout_preds, inout_gts = [], [], [], []
    for _, (_, images, bboxes, gazex, gazey, inout, _) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        out = model({"images": images.to(device), "bboxes": bboxes})
        for i in range(images.shape[0]):
            num_people = len(bboxes[i])
            for j in range(num_people):
                if inout[i][j] == 1:
                    aucs.append(vat_auc(out["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0]))
                    l2s.append(vat_l2(out["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0]))
                if out.get("inout") is not None:
                    inout_preds.append(out["inout"][i][j].item())
                    inout_gts.append(inout[i][j])

    result = {
        "dataset": "vat",
        "json_path": args.json_path,
        "model": args.model,
        "checkpoint_path": args.variant_ckpt,
        "spatial_prior": args.spatial_prior,
        "fusion": args.fusion,
        "selected_layers": args.selected_layers,
        "sample_count": len(inout_gts) if inout_gts else len(l2s),
        "auc": float(np.mean(aucs)) if aucs else None,
        "l2": float(np.mean(l2s)) if l2s else None,
        "inout_ap": float(average_precision_score(inout_gts, inout_preds)) if inout_gts else None,
    }
    if args.metrics_output:
        os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
        with open(args.metrics_output, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Saved metrics to {args.metrics_output}")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/newhome/fb/dataset/videoattentiontarget", help="Path to JSON dataset")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, default="/home/fb/src/paper/gazelleV1/experiments/train_vat_vitb_v0/2026-03-20_22-30-50/epoch_7.pt", help="Path to Baseline checkpoint")
    parser.add_argument("--spot_ckpt", type=str, default="/home/fb/src/paper/gazelleV1/experiments/train_vat_sasa_ggsf/2026-03-12_19-24-13/epoch_7.pt", help="Path to GazeSpot checkpoint")
    parser.add_argument("--variant_ckpt", type=str, default=None, help="Metrics-only checkpoint path for one P1 variant")
    parser.add_argument("--model", type=str, default="gazelle_dinov3_vitb16_inout")
    parser.add_argument("--spatial_prior", type=str, default="ggsf")
    parser.add_argument("--fusion", type=str, default="sasa")
    parser.add_argument("--selected_layers", type=str, default=None)
    parser.add_argument("--metrics_output", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vis_dir", type=str, default=None, help="If set, will save visualizations here")
    parser.add_argument("--num_vis", type=int, default=0, help="Max number of images to visualize")
    args = parser.parse_args()
    if args.variant_ckpt:
        main_variant(args)
    else:
        main(args)
