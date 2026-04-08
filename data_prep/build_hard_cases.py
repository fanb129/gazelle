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

def build_subsets(json_path, output_dir):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    crowd_subset = []
    valid_distances = []

    # 1. 筛选 Crowd 子集 (人数 >= 3)，同时收集所有有效距离
    for item in data:
        # 统计 Crowd (拥挤场景)
        if len(item['heads']) >= 2:
            crowd_subset.append(item)
            
        # 收集有效距离，用于计算 Near 和 Far (对于多人的图，取最短距离作为 Near 的基准，取最远作为 Far 的基准，这里简化为只取第一个人的距离来代表这张图的尺度，或者记录所有的)
        # 为了严谨，我们针对这整张图片的平均注视距离进行排序
        dists = []
        for head in item['heads']:
            dist = calculate_distance(head['bbox_norm'], head['gazex_norm'][0], head['gazey_norm'][0])
            if dist > 0:
                dists.append(dist)
        
        if len(dists) > 0:
            avg_dist = sum(dists) / len(dists)
            valid_distances.append((avg_dist, item))

    # 2. 按距离从小到大排序
    valid_distances.sort(key=lambda x: x[0])
    
    # 3. 截取前 20% (最近) 和后 20% (最远)
    top_20_percent_count = int(len(valid_distances) * 0.20)
    
    near_subset = [item for dist, item in valid_distances[:top_20_percent_count]]
    far_subset = [item for dist, item in valid_distances[-top_20_percent_count:]]

    # 4. 保存文件
    os.makedirs(output_dir, exist_ok=True)
    crowd_path = os.path.join(output_dir, "test_crowd.json")
    near_path = os.path.join(output_dir, "test_near.json")
    far_path = os.path.join(output_dir, "test_far.json")

    with open(crowd_path, 'w') as f:
        json.dump(crowd_subset, f)
    with open(near_path, 'w') as f:
        json.dump(near_subset, f)
    with open(far_path, 'w') as f:
        json.dump(far_subset, f)

    print(f"Total valid original images: {len(valid_distances)}")
    print(f"1. Crowd Subset (>=3 people): {len(crowd_subset)} images saved to {crowd_path}")
    print(f"2. Near Subset (Closest 20%): {len(near_subset)} images saved to {near_path}")
    print(f"3. Far Subset (Farthest 20%): {len(far_subset)} images saved to {far_path}")

if __name__ == "__main__":
    # 请替换为你真实的 json 路径
    JSON_PATH = "/newhome/fb/dataset/gazefollow_extended/test_preprocessed.json"
    OUTPUT_DIR = "/newhome/fb/dataset/gazefollow_extended/test_preprocessed/"
    build_subsets(JSON_PATH, OUTPUT_DIR)