import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as F

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """将归一化的 Tensor 还原为 numpy 图片"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).cpu().numpy()

def plot_gazelle_results(input_data, model_output, index=0, save_path=None):
    """
    可视化 Gazelle 模型的输出，包含 GGSF, SASA 和 Aux Heatmap
    """
    # 1. 获取数据
    img_tensor = input_data["images"][index]
    bboxes = input_data["bboxes"][index]
    
    # 获取计算当前图片人数的索引切分 (很多模块输出是基于 Total_People 的)
    num_ppl_per_img = [len(b) for b in input_data["bboxes"]]
    start_idx = sum(num_ppl_per_img[:index])
    end_idx = start_idx + num_ppl_per_img[index]

    # 获取 Main Heatmap (已经被 split 过了，所以直接取 index)
    main_heatmaps = model_output['heatmap'][index].detach().cpu()
    
    # 获取 Aux Heatmap
    aux_heatmaps = None
    if model_output.get('aux_heatmap') is not None:
        aux_heatmaps = model_output['aux_heatmap'][index].detach().cpu()
        
    # 获取 SASA Layer Weights (未被 split，需要手动切片)
    layer_weights = None
    if model_output.get('layer_weights') is not None:
        layer_weights = model_output['layer_weights'][start_idx:end_idx].detach().cpu().numpy()

    # 【新增】获取 GGSF Geo Mask (未被 split，需要手动切片)
    geo_masks = None
    if model_output.get('geo_mask') is not None:
        # shape 应该是 [num_people_in_this_img, 1, H_feat, W_feat]
        geo_masks = model_output['geo_mask'][start_idx:end_idx].detach().cpu()

    # 还原图片
    img_np = denormalize(img_tensor)
    H, W, _ = img_np.shape

    # 2. 开始绘图
    num_people = len(bboxes)
    if num_people == 0:
        print("No people in this image.")
        return

    # 动态计算需要的列数
    cols = 2 # 至少有 Input 和 Main Heatmap
    if geo_masks is not None: cols += 1
    if aux_heatmaps is not None: cols += 1
    if layer_weights is not None: cols += 1
    
    fig, axes = plt.subplots(num_people, cols, figsize=(4 * cols, 4 * num_people), squeeze=False)

    for i in range(num_people):
        bbox = bboxes[i] 
        col_idx = 0
        
        # --- Column 1: 原图 + Head Bbox ---
        ax = axes[i, col_idx]
        ax.imshow(img_np)
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            rect = patches.Rectangle(
                (xmin * W, ymin * H), (xmax - xmin) * W, (ymax - ymin) * H,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
        ax.set_title(f"Person {i+1} Input")
        ax.axis('off')
        col_idx += 1

        # --- 【新增】 Column 2: GGSF Geometric Mask ---
        if geo_masks is not None:
            ax = axes[i, col_idx]
            g_mask = geo_masks[i].squeeze(0).numpy() # [H_feat, W_feat]
            
            # 【增加这一步：拉伸对比度】
            g_mask_min = g_mask.min()
            g_mask_max = g_mask.max()
            # 只有当最大最小值有明显差异时才拉伸，避免除以 0
            if g_mask_max - g_mask_min > 1e-5:
                g_mask = (g_mask - g_mask_min) / (g_mask_max - g_mask_min)
            
            # 缩放回原图大小
            g_mask_img = F.to_pil_image(torch.from_numpy(g_mask)).resize((W, H), Image.BILINEAR)
            g_mask_np = np.array(g_mask_img) # 此时已经是 0-1 之间了
            
            ax.imshow(img_np)
            ax.imshow(g_mask_np, cmap='plasma', alpha=0.6) 
            ax.set_title("GGSF Spatial Gate")
            ax.axis('off')
            col_idx += 1

        # --- Column 3: Main Heatmap ---
        ax = axes[i, col_idx]
        hm = main_heatmaps[i]
        hm_img = F.to_pil_image(hm).resize((W, H), resample=Image.BILINEAR)
        hm_np = np.array(hm_img)
        
        ax.imshow(img_np)
        ax.imshow(hm_np, cmap='jet', alpha=0.5)
        y_max, x_max = np.unravel_index(hm_np.argmax(), hm_np.shape)
        ax.plot(x_max, y_max, 'w+', markersize=15, markeredgewidth=3)
        ax.set_title("Main Prediction")
        ax.axis('off')
        col_idx += 1
        
        # --- Column 4: Aux Heatmap (Optional) ---
        if aux_heatmaps is not None:
            ax = axes[i, col_idx]
            hm_aux = aux_heatmaps[i]
            hm_aux_img = F.to_pil_image(hm_aux).resize((W, H), Image.BILINEAR)
            hm_aux_np = np.array(hm_aux_img)
            
            ax.imshow(img_np)
            ax.imshow(hm_aux_np, cmap='viridis', alpha=0.5)
            ax.set_title("Aux Prediction (Layer 4)")
            ax.axis('off')
            col_idx += 1

        # --- Column 5: SASA Weights (Optional) ---
        if layer_weights is not None:
            ax = axes[i, col_idx]
            weights = layer_weights[i] # [4]
            # 这里可以根据传入的 model 自动判断，或者写死
            layers = ['L_Shallow', 'L_Mid1', 'L_Mid2', 'L_Deep']
            if save_path and "vitl" in save_path:
                layers = ['L5', 'L11', 'L17', 'L23']
            
            bars = ax.bar(layers, weights, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylim(0, 1.0)
            ax.set_title("SASA Layer Attention")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
            col_idx += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()