import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载预训练模型
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def detect_day_night(image_path):
    """检测图片是白天还是黑夜"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])
    return 'day' if brightness > 127 else 'night'

def extract_features(image_path):
    """提取混合特征"""
    # 传统特征
    img = Image.open(image_path).resize((64, 64)).convert('L')
    traditional_feat = np.array(img).flatten()
    
    # 深度特征
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img.convert('RGB'))
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))
    deep_feat = feature_extractor.predict(img_array).flatten()
    
    # 特征融合
    return np.concatenate([traditional_feat/np.linalg.norm(traditional_feat), 
                          deep_feat/np.linalg.norm(deep_feat)])

def find_similar_pairs(image_dir):
    """主处理函数"""
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
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
    np.fill_diagonal(sim_matrix, -1)
    
    paired = set()
    pairs = []
    for i in range(len(valid_paths)):
        if i in paired:
            continue
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
