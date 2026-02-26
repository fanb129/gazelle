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
    可视化 Gazelle 模型的输出
    Args:
        input_data: dict, 包含 'images' (Tensor [B,3,H,W]) 和 'bboxes' (List of Lists)
        model_output: dict, 模型 forward 的返回值
        index: int, 要可视化 Batch 中的第几张图片
        save_path: str, 如果不为None，保存图片到该路径
    """
    # 1. 获取数据
    img_tensor = input_data["images"][index]
    bboxes = input_data["bboxes"][index]
    
    # 获取预测结果 (注意：model_output['heatmap'] 是一个 list，对应每个 batch)
    # heatmap_preds[index] 是 shape 为 [N_people, 64, 64] 的 tensor
    main_heatmaps = model_output['heatmap'][index].detach().cpu()
    
    # 获取 Aux Heatmap (如果有)
    aux_heatmaps = None
    if model_output.get('aux_heatmap') is not None:
        aux_heatmaps = model_output['aux_heatmap'][index].detach().cpu()
        
    # 获取 Layer Weights (如果有)
    # layer_weights 通常是 [Total_People, 4]，没有被 split_tensors
    # 我们需要手动切分找到对应当前图片的人的权重
    layer_weights = None
    if model_output.get('layer_weights') is not None:
        # 计算当前图片之前有多少人
        num_ppl_per_img = [len(b) for b in input_data["bboxes"]]
        start_idx = sum(num_ppl_per_img[:index])
        end_idx = start_idx + num_ppl_per_img[index]
        layer_weights = model_output['layer_weights'][start_idx:end_idx].detach().cpu().numpy()

    # 还原图片
    img_np = denormalize(img_tensor)
    H, W, _ = img_np.shape

    # 2. 开始绘图
    num_people = len(bboxes)
    if num_people == 0:
        print("No people in this image.")
        return

    # 设定画布：每行显示一个人，每列显示不同信息
    # 列定义: [Original+Bbox, Main Heatmap, Aux Heatmap, Layer Weights]
    cols = 4 if (aux_heatmaps is not None and layer_weights is not None) else 2
    if aux_heatmaps is not None: cols = max(cols, 3)
    if layer_weights is not None: cols = max(cols, 3)
    
    fig, axes = plt.subplots(num_people, cols, figsize=(4 * cols, 4 * num_people), squeeze=False)

    for i in range(num_people):
        bbox = bboxes[i] # (xmin, ymin, xmax, ymax) normalized
        
        # --- Column 1: 原图 + Head Bbox ---
        ax = axes[i, 0]
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

        # --- Column 2: Main Heatmap ---
        ax = axes[i, 1]
        hm = main_heatmaps[i] # [64, 64]
        # Resize heatmap to image size for overlay
        hm_img = F.to_pil_image(hm)
        hm_img = hm_img.resize((W, H), resample=Image.BILINEAR)
        hm_np = np.array(hm_img)
        
        ax.imshow(img_np)
        ax.imshow(hm_np, cmap='jet', alpha=0.5) # 叠加显示
        
        # 画最大值点
        y_max, x_max = np.unravel_index(hm_np.argmax(), hm_np.shape)
        ax.plot(x_max, y_max, 'w+', markersize=15, markeredgewidth=3) # 白色十字
        
        ax.set_title("Main Prediction")
        ax.axis('off')

        col_idx = 2
        
        # --- Column 3: Aux Heatmap (Optional) ---
        if aux_heatmaps is not None:
            ax = axes[i, col_idx]
            hm_aux = aux_heatmaps[i]
            hm_aux_img = F.to_pil_image(hm_aux).resize((W, H), Image.BILINEAR)
            hm_aux_np = np.array(hm_aux_img)
            
            ax.imshow(img_np)
            ax.imshow(hm_aux_np, cmap='viridis', alpha=0.5) # 用不同颜色区分
            ax.set_title("Aux Prediction (Layer 4)")
            ax.axis('off')
            col_idx += 1

        # --- Column 4: SASA Weights (Optional) ---
        if layer_weights is not None:
            ax = axes[i, col_idx]
            weights = layer_weights[i] # [4]
            if "vitl" in save_path:
                layers = ['L5 (Shallow)', 'L11', 'L17', 'L23 (Deep)']
            elif "vitb" in save_path:
                layers = ['L2 (Shallow)', 'L5', 'L8', 'L11 (Deep)']
            
            bars = ax.bar(layers, weights, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylim(0, 1.0)
            ax.set_title("SASA Layer Attention")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 在柱子上标数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()