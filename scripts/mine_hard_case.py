import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm

# 导入你的模型工厂和评估工具
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_l2

# ==========================================
# 数据集类 (用于加载数据和返回路径)
# ==========================================
class GazeFollow(torch.utils.data.Dataset):
    def __init__(self, data_path, json_path, img_transform):
        self.data_path = data_path
        self.images = json.load(open(json_path, "rb"))
        self.transform = img_transform

    def __getitem__(self, idx):
        item = self.images[idx]
        img_rel_path = item['path']
        img_abs_path = os.path.join(self.data_path, img_rel_path)
        
        image = self.transform(Image.open(img_abs_path).convert("RGB"))
        bboxes = [head['bbox_norm'] for head in item['heads']]
        gazex = [head['gazex_norm'] for head in item['heads']]
        gazey = [head['gazey_norm'] for head in item['heads']]
        
        return image, bboxes, gazex, gazey, img_abs_path

    def __len__(self):
        return len(self.images)

def collate(batch):
    images, bboxes, gazex, gazey, paths = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(paths)

# ==========================================
# 主函数
# ==========================================
@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    print("Loading GazeSpot Model to mine hard cases...")
    # 启用完整的 GazeSpot 架构
    model, transform = get_gazelle_model(args.model_name, use_sasa=True, use_ggsf=True, use_aux=False)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, map_location="cpu", weights_only=True))
    model.to(device).eval()

    dataset = GazeFollow(args.data_path, args.json_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    # 用于保存所有预测结果的误差
    error_records = []

    for _, (images, bboxes, gazex, gazey, paths) in tqdm(enumerate(dataloader), desc="Mining Limitations", total=len(dataloader)):
        out_spot = model({"images": images.to(device), "bboxes": bboxes})
        
        for i in range(images.shape[0]):
            num_people = len(bboxes[i])
            img_path = paths[i]
            
            # 记录一张图中误差最大的那个人
            max_l2_in_image = -1.0
            
            for j in range(num_people):
                # 兼容处理 GT
                gx = gazex[i][j][0] if isinstance(gazex[i][j], list) else gazex[i][j]
                gy = gazey[i][j][0] if isinstance(gazey[i][j], list) else gazey[i][j]
                
                # 如果有有效的 GT
                if gx >= 0 and gy >= 0:
                    # 获取该预测的 L2 误差 (gazefollow_l2 返回 avg_l2, min_l2)
                    _, min_l2 = gazefollow_l2(out_spot['heatmap'][i][j], gx, gy)
                    if min_l2 > max_l2_in_image:
                        max_l2_in_image = min_l2
            
            # 如果这张图里有有效预测，保存它的最大误差和路径
            if max_l2_in_image >= 0:
                error_records.append({
                    "path": img_path,
                    "error": max_l2_in_image
                })

    # ==========================================
    # 按照 L2 误差从大到小排序
    # ==========================================
    print("\nSorting cases by L2 error (descending)...")
    error_records.sort(key=lambda x: x["error"], reverse=True)

    # 提取前 top_k 张误差最大的图片路径 (利用 Set 去重)
    top_k_paths = []
    seen_paths = set()
    
    for record in error_records:
        if record["path"] not in seen_paths:
            top_k_paths.append(record["path"])
            seen_paths.add(record["path"])
        if len(top_k_paths) >= args.top_k:
            break

    # ==========================================
    # 写入到 .txt 文件
    # ==========================================
    os.makedirs(os.path.dirname(args.output_txt) if os.path.dirname(args.output_txt) else '.', exist_ok=True)
    with open(args.output_txt, "w") as f:
        for path in top_k_paths:
            f.write(f"{path}\n")

    print("=" * 50)
    print(f"✅ Successfully mined top {args.top_k} worst failure cases!")
    print(f"Top 1 worst L2 Error: {error_records[0]['error']:.4f}")
    print(f"Top {args.top_k} worst L2 Error: {error_records[args.top_k-1]['error']:.4f}")
    print(f"File saved to: {args.output_txt}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/newhome/fb/dataset/gazefollow_extended", help="数据集根目录")
    parser.add_argument("--json_path", type=str, default="/newhome/fb/dataset/gazefollow_extended/test_preprocessed.json", help="对应的 JSON 标注文件")
    parser.add_argument("--ckpt_path", type=str, default="/home/fb/src/paper/gazelleV1/experiments/train_gazefollow_sasa_ggsf/2026-02-26_15-51-06/epoch_14.pt", help="GazeSpot 权重路径")
    parser.add_argument("--model_name", type=str, default="gazelle_dinov3_vitb16", help="使用的模型名字")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=40, help="要挖掘并保存的失败案例数量")
    parser.add_argument("--output_txt", type=str, default="/home/fb/src/paper/gazelleV1/scripts/real_hard_cases.txt", help="生成的路径列表文件")
    
    args = parser.parse_args()
    main(args)