import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ==========================================
# 视觉美化工具函数
# ==========================================
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy()

transform_base = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 数据集重写 (均匀等距采样)
# ==========================================
class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, json_path, transform_base, num_vis):
        print(f"Loading annotations from {json_path}...")
        self.sequences = json.load(open(os.path.join(path, json_path), "rb"))
        self.path = path
        self.transform_base = transform_base
        
        all_valid_frames = []
        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i]['frames'])):
                frame = self.sequences[i]['frames'][j]
                has_valid_person = False
                for head in frame.get('heads', []):
                    if head.get('inout', 1) == 1:
                        has_valid_person = True
                        break
                if has_valid_person:
                    all_valid_frames.append((i, j))
        
        total_valid = len(all_valid_frames)
        if total_valid > num_vis:
            step = total_valid // num_vis
            self.frames = all_valid_frames[::step][:num_vis]
        else:
            self.frames = all_valid_frames
            
        print(f"Dataset initialized: Sampled {len(self.frames)} diverse frames out of {total_valid} valid ones.")

    def __getitem__(self, idx):
        seq = self.sequences[self.frames[idx][0]]
        frame = seq['frames'][self.frames[idx][1]]
        pil_img = Image.open(os.path.join(self.path, frame['path'])).convert("RGB")
        
        img_base = self.transform_base(pil_img)
        
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]
        return img_base, bboxes, gazex, gazey, inout, frame['path']

    def __len__(self): 
        return len(self.frames)
    
def collate(batch):
    img_base, bboxes, gazex, gazey, inout, paths = zip(*batch)
    return torch.stack(img_base), list(bboxes), list(gazex), list(gazey), list(inout), list(paths)

@torch.no_grad()
def main(args):
    device = "cpu"
    print(f"Running on {device}")
    
    if args.vis_dir: os.makedirs(args.vis_dir, exist_ok=True)

    dataset = VideoAttentionTarget(args.data_path, args.json_path, transform_base, args.num_vis)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    saved_vis = 0

    for _, (images_base, bboxes, gazex, gazey, inout, paths) in tqdm(enumerate(dataloader), desc="Generating Images", total=len(dataloader)):        
        for i in range(images_base.shape[0]): 
            num_people = len(bboxes[i])

            if args.vis_dir and saved_vis < args.num_vis and num_people > 0 and 1 in inout[i]:
                
                img_np = denormalize(images_base[i])
                img_h, img_w = img_np.shape[:2]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img_np)
                ax.axis('off')
                
                # 遍历图中的人
                for j in range(num_people):
                    # 找到第一个看向画面内的人
                    if inout[i][j] == 1:
                        bbox = bboxes[i][j]
                        x_gt = gazex[i][j][0] if isinstance(gazex[i][j], list) else gazex[i][j]
                        y_gt = gazey[i][j][0] if isinstance(gazey[i][j], list) else gazey[i][j]
                        
                        if bbox is not None and x_gt >= 0 and y_gt >= 0:
                            xmin, ymin, xmax, ymax = bbox
                            xmin, ymin, xmax, ymax = xmin * img_w, ymin * img_h, xmax * img_w, ymax * img_h
                            
                            x_center = (xmin + xmax) / 2.0
                            y_center = (ymin + ymax) / 2.0
                            
                            gt_x = x_gt * img_w
                            gt_y = y_gt * img_h
                            
                            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                                     linewidth=4, edgecolor='#00ff00', facecolor='none')
                            ax.add_patch(rect)
                            
                            ax.plot([x_center, gt_x], [y_center, gt_y], color='#00ff00', linewidth=3, linestyle='-')
                            ax.plot(gt_x, gt_y, marker='*', color='#00ff00', markersize=20, markeredgecolor='white', markeredgewidth=2)
                        
                        # 【核心修改】：画完这一个人之后，直接 break，不再画其他人！
                        break 

                plt.tight_layout(pad=0)
                img_name = paths[i].replace("/", "_")
                plt.savefig(os.path.join(args.vis_dir, f"new_vis_{img_name}"), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                saved_vis += 1

            if saved_vis >= args.num_vis:
                print(f"\n✅ Finished generating {args.num_vis} unique images with single bbox!")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/newhome/fb/dataset/videoattentiontarget", help="Path to JSON dataset")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vis_dir", type=str, required=True, help="If set, will save visualizations here")
    parser.add_argument("--num_vis", type=int, default=100, help="Max number of images to visualize")
    args = parser.parse_args()
    main(args)