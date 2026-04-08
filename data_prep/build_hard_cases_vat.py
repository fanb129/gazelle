import json
import math
import os

def calculate_distance(bbox, gx, gy):
    """计算人头中心点到真实注视点的归一化欧氏距离"""
    if gx < 0 or gy < 0 or bbox is None:
        return -1
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return math.sqrt((cx - gx)**2 + (cy - gy)**2)

def build_vat_subsets(json_path, output_dir):
    print(f"Loading VAT annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    crowd_frames = []
    valid_distance_frames = []
    total_frames = 0

    # 1. 遍历 Sequence 和 Frames
    for seq in data:
        for frame in seq['frames']:
            total_frames += 1
            
            # 【筛选条件 1】: Crowd (拥挤场景)
            # VAT 中人比较多，我们将阈值设为 >= 4，极具挑战性！
            if len(frame['heads']) > 4:
                crowd_frames.append(frame)
                
            # 【筛选条件 2】: 距离尺度 (Near / Far)
            dists = []
            for head in frame['heads']:
                # VAT 特有：只有在看画面内 (inout == 1) 的目标，计算距离才有物理意义
                if head.get('inout', 1) == 1:
                    # 兼容处理：VAT的注视点有时是一个列表 [x]
                    gx = head['gazex_norm'][0] if isinstance(head['gazex_norm'], list) else head['gazex_norm']
                    gy = head['gazey_norm'][0] if isinstance(head['gazey_norm'], list) else head['gazey_norm']
                    
                    dist = calculate_distance(head['bbox_norm'], gx, gy)
                    if dist > 0:
                        dists.append(dist)
            
            # 如果这张图里有在看画面内的人，计算他们的平均注视距离
            if len(dists) > 0:
                avg_dist = sum(dists) / len(dists)
                valid_distance_frames.append((avg_dist, frame))

    # 2. 按距离从小到大排序
    valid_distance_frames.sort(key=lambda x: x[0])
    
    # 3. 截取前 20% (最近) 和后 20% (最远)
    top_20_percent_count = int(len(valid_distance_frames) * 0.20)
    
    near_frames = [f for d, f in valid_distance_frames[:top_20_percent_count]]
    far_frames = [f for d, f in valid_distance_frames[-top_20_percent_count:]]

    # ==========================================
    # 【新增】4. 混合并去重，生成终极困难子集 (Mixed Hard Cases)
    # ==========================================
    # 使用字典来去重，键为图像的唯一路径，值为图像帧数据
    mixed_dict = {}
    for frame in (crowd_frames + near_frames + far_frames):
        mixed_dict[frame['path']] = frame
        
    mixed_frames = list(mixed_dict.values())

    # 5. 完美伪装回 VAT 的原始 JSON 结构 [{"frames": [...]}]
    # 这样 eval_vat.py 就不需要做任何修改，直接能读！
    crowd_data = [{"frames": crowd_frames}]
    near_data = [{"frames": near_frames}]
    far_data = [{"frames": far_frames}]
    mixed_data = [{"frames": mixed_frames}]  # 混合数据的包装

    # 6. 保存文件
    os.makedirs(output_dir, exist_ok=True)
    crowd_path = os.path.join(output_dir, "test_crowd_gt4.json")
    near_path = os.path.join(output_dir, "test_near.json")
    far_path = os.path.join(output_dir, "test_far.json")
    mixed_path = os.path.join(output_dir, "test_mixed_hard.json")  # 混合数据的文件名

    with open(crowd_path, 'w') as f:
        json.dump(crowd_data, f)
    with open(near_path, 'w') as f:
        json.dump(near_data, f)
    with open(far_path, 'w') as f:
        json.dump(far_data, f)
    with open(mixed_path, 'w') as f:
        json.dump(mixed_data, f)

    print(f"\n✅ Processing Complete! Processed {total_frames} frames in total.")
    print(f"1. Crowd Subset (>=4 people): {len(crowd_frames):>5} frames saved to {crowd_path}")
    print(f"2. Near Subset (Closest 20%): {len(near_frames):>5} frames saved to {near_path}")
    print(f"3. Far Subset (Farthest 20%): {len(far_frames):>5} frames saved to {far_path}")
    print("-" * 70)
    print(f"🔥 Mixed Hard Subset (Deduplicated): {len(mixed_frames):>5} frames saved to {mixed_path}")
    print("-" * 70)

if __name__ == "__main__":
    # 请替换为你真实的 VAT json 路径
    JSON_PATH = "/newhome/fb/dataset/videoattentiontarget/test_preprocessed.json"
    OUTPUT_DIR = "/newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets"
    build_vat_subsets(JSON_PATH, OUTPUT_DIR)

'''
Loading VAT annotations from /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json...

✅ Processing Complete! Processed 13127 frames in total.
1. Crowd Subset (>=4 people):  2167 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd.json
2. Near Subset (Closest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_near.json
3. Far Subset (Farthest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json
----------------------------------------------------------------------
🔥 Mixed Hard Subset (Deduplicated):  5724 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json
----------------------------------------------------------------------

✅ Processing Complete! Processed 13127 frames in total.
1. Crowd Subset (>=3 people):  5182 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_3.json
2. Near Subset (Closest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_near.json
3. Far Subset (Farthest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json
----------------------------------------------------------------------
🔥 Mixed Hard Subset (Deduplicated):  7575 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json


(py310) fb@nise-server:~/src/paper/gazelleV1$ /home/fb/anaconda3/envs/py310/bin/python /home/fb/src/paper/gazelleV1/data_prep/build_hard_cases_vat.py
Loading VAT annotations from /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json...

✅ Processing Complete! Processed 13127 frames in total.
1. Crowd Subset (==3 people):  3015 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_eq3.json
2. Near Subset (Closest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_near.json
3. Far Subset (Farthest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json
----------------------------------------------------------------------
🔥 Mixed Hard Subset (Deduplicated):  6007 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json
---------

(py310) fb@nise-server:~/src/paper/gazelleV1$ /home/fb/anaconda3/envs/py310/bin/python /home/fb/src/paper/gazelleV1/data_prep/build_hard_cases_vat.py
Loading VAT annotations from /newhome/fb/dataset/videoattentiontarget/test_preprocessed.json...

✅ Processing Complete! Processed 13127 frames in total.
1. Crowd Subset (==4 people):  1628 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_eq4.json
2. Near Subset (Closest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_near.json
3. Far Subset (Farthest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json
----------------------------------------------------------------------
🔥 Mixed Hard Subset (Deduplicated):  5444 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json

✅ Processing Complete! Processed 13127 frames in total.
1. Crowd Subset (<3 people):  7945 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_lt3.json
2. Near Subset (Closest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_near.json
3. Far Subset (Farthest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json
----------------------------------------------------------------------
🔥 Mixed Hard Subset (Deduplicated):  9708 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json


✅ Processing Complete! Processed 13127 frames in total.
1. Crowd Subset (>4 people):   539 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_crowd_gt4.json
2. Near Subset (Closest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_near.json
3. Far Subset (Farthest 20%):  2078 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_far.json
----------------------------------------------------------------------
🔥 Mixed Hard Subset (Deduplicated):  4436 frames saved to /newhome/fb/dataset/videoattentiontarget/test_preprocessed_subsets/test_mixed_hard.json



'''