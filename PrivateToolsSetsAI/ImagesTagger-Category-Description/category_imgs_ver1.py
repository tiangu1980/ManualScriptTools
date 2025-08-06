
import os
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from collections import Counter

def load_image_features(image_path, model):
    """提取ResNet50特征并展平为一维数组"""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()  # 关键修改：2048维特征展平^^1^^

def get_color_features(image_path):
    """提取RGB三通道均值特征"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = np.array(img)
    return np.mean(pixels, axis=(0, 1)).flatten()  # 确保输出1D数组^^6^^

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '--path', dest='path', required=True, help='文件夹路径')
    parser.add_argument('--c', '--category', dest='category', 
                       choices=['landscape', 'color', 'animal'], required=True, help='分类维度')
    parser.add_argument('--m', '--min_count', dest='min_count', type=int, default=20, help='最小分类图片数量')
    args = parser.parse_args()

    model = ResNet50(weights='imagenet')
    features_list = []
    valid_paths = []

    for root, _, files in os.walk(args.path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    features = get_color_features(img_path) if args.category == 'color' \
                              else load_image_features(img_path, model)
                    features_list.append(features)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"跳过损坏文件 {img_path}: {str(e)}")

    if len(features_list) < args.min_count:
        raise ValueError(f"有效图片数量不足{args.min_count}张")

    # 关键修改：确保特征矩阵为(n_samples, n_features)格式^^1^^5^^
    features_array = np.vstack(features_list)  
    print("features_array结构:", type(features_array), len(features_array))
    optimal_clusters = max(1, min(10, len(features_array)//args.min_count))
    kmeans = KMeans(n_clusters=optimal_clusters)
    #kmeans = KMeans(n_clusters=min(10, len(features_array)//5))
    clusters = kmeans.fit_predict(features_array)

    # 结果输出
    print("Category    File")
    for i, (path, cluster) in enumerate(zip(valid_paths, clusters)):
        print(f"{cluster}    {os.path.basename(path)}")
    
    # 统计每个cluster的元素数量
    cluster_counts = Counter(clusters)
    
    # 按数量降序排列并打印
    sorted_counts = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    for cluster, count in sorted_counts:
        print(f"{cluster}    {count}")

if __name__ == '__main__':
    main()
