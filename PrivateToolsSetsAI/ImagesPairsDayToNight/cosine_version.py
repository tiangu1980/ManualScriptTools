import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tensorflow.keras.applications import MobileNetV2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误信息

def detect_day_night(image_path):
    """检测图片是白天还是黑夜"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])
    return 'day' if brightness > 127 else 'night'

def extract_features(image_path):
    """提取图片特征向量"""
    img = Image.open(image_path).resize((64, 64)).convert('L')
    return np.array(img).flatten()

def find_similar_pairs(image_dir):
    """主处理函数"""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 提取特征和昼夜信息
    features = []
    day_night = []
    valid_paths = []
    for path in image_paths:
        dn = detect_day_night(path)
        if dn:
            features.append(extract_features(path))
            day_night.append(dn)
            valid_paths.append(path)
    
    # 计算相似度矩阵
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, -1)  # 排除自身
    
    # 配对处理
    paired = set()
    pairs = []
    for i in range(len(valid_paths)):
        if i in paired:
            continue
        # 找最相似且昼夜相反的图片
        candidates = [(j, sim_matrix[i][j]) for j in range(len(valid_paths))
                     if j not in paired and day_night[i] != day_night[j]]
        if not candidates:
            continue
        best_match = max(candidates, key=lambda x: x[1])
        j = best_match[0]
        paired.update({i, j})
        pairs.append((valid_paths[i], valid_paths[j], best_match[1]))
    
    return pairs

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_matcher.py <image_directory>")
        sys.exit(1)
    
    result = find_similar_pairs(sys.argv[1])
    for pair in result:
        print(f"Pair: {pair[0]} (day) <-> {pair[1]} (night) | Similarity: {pair[2]:.2f}")
