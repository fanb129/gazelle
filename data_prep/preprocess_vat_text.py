import json
import torch
import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === 1. 加载 VLM 模型 (以 Qwen-VL-Chat 为例) ===
# 显存优化：确保你安装了 auto-gptq 或 bitsandbytes 以支持 Int4
MODEL_ID = "Qwen/Qwen-VL-Chat-Int4" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True).eval()

# === 2. 配置路径 ===
DATA_DIR = "/newhome/fb/dataset/videoattentiontarget" # 修改为你的路径

# 定义要处理的数据集分片
SPLITS = ["train", "test"] 

# === 3. 辅助函数：扩大 BBox ===
def get_expanded_bbox(bbox, img_w, img_h):
    """
    将头部 bbox 扩大以包含身体区域，便于识别衣着。
    策略：左右适度扩大，下方大幅扩大（看身体），上方少量扩大。
    """
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    
    # 扩大系数 (你可以根据实际效果调整)
    # 左右扩大 1.5倍 (w * 0.25 * 2 + w)
    # 上方扩大 1.2倍
    # 下方扩大 3.0倍 (为了看到衣服)
    
    new_xmin = max(0, xmin - w * 0.5)
    new_xmax = min(img_w, xmax + w * 0.5)
    
    new_ymin = max(0, ymin - h * 0.2)
    # 重点：头部下方延伸更多，以包含躯干/衣服
    new_ymax = min(img_h, ymax + h * 3.0) 
    
    return [int(new_xmin), int(new_ymin), int(new_xmax), int(new_ymax)]

# === 4. 核心处理循环 ===
for split in SPLITS:
    print(f"========== Processing {split} set ==========")
    
    json_filename = f"{split}_preprocessed.json"
    output_filename = f"{split}_preprocessed_text.json"
    
    json_path = os.path.join(DATA_DIR, json_filename)
    output_path = os.path.join(DATA_DIR, output_filename)
    
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skipping...")
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)

    new_data = []
    
    # 遍历每一张图
    # 注意：正式运行时请去掉切片 [:10]
    for i, item in enumerate(tqdm(data, desc=f"Generating {split}")):
        image_path = os.path.join(DATA_DIR, item['path'])
        
        # 必须读取图片尺寸用于 bbox 边界检查 (clip)
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue

        new_heads = []
        for head in item['heads']:
            raw_bbox = head['bbox'] # 原始头部 bbox
            
            # --- 关键修改：获取扩大的 bbox (Head -> Upper Body) ---
            body_bbox = get_expanded_bbox(raw_bbox, img_w, img_h)
            
            # 构造 Query
            # 提示词微调：明确告诉模型我们给的是大致区域，让它描述区域里的人
            # Qwen-VL 对于坐标的理解能力较强，直接给坐标数字通常能懂，
            # 如果效果不好，可以使用 Qwen 特有的 <box> 格式，但自然语言通常足够。
            prompt = (
                f"Identify the person located roughly at coordinates {body_bbox}. "
                "Describe their clothing color and type (e.g., 'red t-shirt', 'blue jacket') "
                "and gender in a short phrase. Keep it under 10 words."
            )

            query = tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt}
            ])
            
            # 推理
            try:
                with torch.no_grad():
                    response, _ = model.chat(tokenizer, query=query, history=None)
                
                # 清洗文本
                text_label = response.strip().rstrip('.')
                
                # 存入数据
                head['text_label'] = text_label
                # 也可以选择把 body_bbox 也存下来用于 debug，看模型到底看了哪里
                # head['body_bbox_debug'] = body_bbox 
                
            except Exception as e:
                print(f"Inference error: {e}")
                head['text_label'] = "person" # fallback

            new_heads.append(head)
        
        item['heads'] = new_heads
        new_data.append(item)

        # 定期保存 (每1000张存一次，或者每处理完10%存一次)
        if (i + 1) % 1000 == 0:
            print(f"Saving checkpoint to {output_path}...")
            with open(output_path, 'w') as f:
                json.dump(new_data, f)
        
        # 测试，只处理第一张
        # break
    # === 5. 该 Split 处理完毕，最终保存 ===
    with open(output_path, 'w') as f:
        json.dump(new_data, f)
    
    print(f"Finished {split}! Saved to {output_path}")

print("All Done!")