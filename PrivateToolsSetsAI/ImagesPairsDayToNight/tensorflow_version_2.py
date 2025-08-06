import os
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载预训练模型并截取特征提取层
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def analyze_day_night_tensor(img_tensor):
    rgb_tensor = img_tensor[..., :3]  # 保留RGB通道，丢弃Alpha
    """使用张量运算分析昼夜相对关系"""
    hsv = tf.image.rgb_to_hsv(rgb_tensor)
    brightness = tf.reduce_mean(hsv[..., 2])
    return tf.cond(brightness > 0.5, lambda: 'day', lambda: 'night')

def analyze_night_day(img_path):
    """分析单张图像的日夜特征"""
    img = cv2.imread(img_path)
    if img is None:
        return {'is_night': False, 'brightness': 0, 'blue_ratio': 0}
    
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 计算全局亮度特征
    brightness = np.mean(hsv[:,:,2]) / 255.0
    
    # 计算蓝色通道占比（夜晚天空通常偏蓝）
    blue_ratio = np.mean(img[:,:,0]) / (np.mean(img) + 1e-6)
    
    # 计算对比度（白天通常对比度更高）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    
    # 综合判断（可调整阈值）
    is_night = (brightness < 0.4) or (blue_ratio > 1.2 and brightness < 0.6)
    
    return {
        'is_night': is_night,
        'brightness': brightness,
        'blue_ratio': blue_ratio,
        'contrast': contrast
    }

def extract_features(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = preprocess_input(np.array(img.convert('RGB')))
    return feature_extractor.predict(np.expand_dims(img_array, axis=0)).flatten()

def find_similar_pairs(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    features = []
    valid_paths = []
    for path in image_paths:
        try:
            features.append(extract_features(path))
            valid_paths.append(path)
        except:
            continue
    
    features = np.array(features)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / norm
    sim_matrix = np.dot(normalized, normalized.T)
    np.fill_diagonal(sim_matrix, -1)
    
    paired = set()
    pairs = []
    for i in range(len(valid_paths)):
        if i in paired:
            continue
        candidates = [(j, sim_matrix[i][j]) for j in range(len(valid_paths)) if j not in paired]
        if not candidates:
            continue
        best_match = max(candidates, key=lambda x: x[1])
        j = best_match[0]
        
        # 张量分析昼夜相对关系
        img1 = tf.convert_to_tensor(np.array(Image.open(valid_paths[i]).resize((224, 224)))/255.0)
        img2 = tf.convert_to_tensor(np.array(Image.open(valid_paths[j]).resize((224, 224)))/255.0)
        dn1 = analyze_day_night_tensor(img1)
        dn2 = analyze_day_night_tensor(img2)
        
        paired.update({i, j})
        pairs.append({
            'day_image': valid_paths[i] if dn1 == 'day' else valid_paths[j],
            'night_image': valid_paths[j] if dn1 == 'day' else valid_paths[i],
            'similarity': best_match[1],
            'confidence': abs(tf.reduce_mean(img1) - tf.reduce_mean(img2)).numpy()
        })
        
    # 原有特征提取代码保持不变...
    # 新增日夜关系验证
    for pair in pairs:
        img1_features = analyze_night_day(pair['day_image'])
        img2_features = analyze_night_day(pair['night_image'])
        
        # 验证日夜关系是否正确
        #if img1_features['is_night'] and not img2_features['is_night']:
        if (img1_features['is_night'] and not img2_features['is_night']) or (img1_features['brightness'] < img2_features['brightness']):
            # 如果发现相反则交换
            pair['day_image'], pair['night_image'] = pair['night_image'], pair['day_image']
        
        # 添加特征数据用于调试
        pair.update({
            'day_features': img1_features,
            'night_features': img2_features
        })
        
    # return pairs, also include the day image and night image day_features and night_features
    # return [(x['day_image'], x['night_image'], x['similarity'], x) for x in pairs]
    return [(x['day_image'], x['night_image'], x['similarity'], x['day_features'], x['night_features'], x) for x in pairs]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_matcher.py <image_directory>")
        sys.exit(1)
        
    print(f"Day\t\tNight\t\tSimilarity")
    result = find_similar_pairs(sys.argv[1])
    orderedResult = sorted(result, key=lambda x: x[2], reverse=True)
    for pair in orderedResult:
        #print(f"Pair: {pair[0]} (day) <-> {pair[1]} (night) | Similarity: {pair[2]:.2f}")
        print(f"{pair[0]}\tday\t{pair[1]}\tnight\t{pair[2]:.2f}")
        #print(f"Pair: {pair[0]} (day) <-> {pair[1]} (night) | Similarity: {pair[2]:.2f} | Day Features: {pair[3]} | Night Features: {pair[4]}")
    
    # for pair in result:
    #     print(f"Day: {pair['day_image']}")
    #     print(f"Night: {pair['night_image']}")
    #     print(f"Similarity: {pair['similarity']:.2f}")
    #     print("-" * 50)
    