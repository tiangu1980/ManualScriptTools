
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
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

def export_clusters_to_excel(valid_paths, clusters, output_path, category_val='color', n_clusters_val=10, tol_val=1e-4):
    # 准备数据
    data = {
        "Category": clusters,
        "File": [os.path.basename(path) for path in valid_paths]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 获取当前时间戳(精确到秒)
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 获取当前时间戳(精确到毫秒)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # 获取文件夹名并构建输出路径
    folder_name = os.path.basename(output_path.rstrip(os.sep))
    excel_path = os.path.join(output_path, f"{folder_name}_({category_val}_n={n_clusters_val}_t={tol_val})_{timestamp}.xlsx").replace("/", "\\")

    # 写入Excel
    df.to_excel(excel_path, index=False)
    print(f"结果已保存至: {excel_path}")

def get_color_features(image_path):
    """提取RGB三通道均值特征"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = np.array(img)
    return np.mean(pixels, axis=(0, 1)).flatten()  # 确保输出1D数组^^6^^

def get_clusters(features_list, n_clusters, tol=1e-4):
    """使用KMeans聚类"""
    kmeans = KMeans(n_clusters=n_clusters, tol=tol)
    return kmeans.fit_predict(features_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '--path', dest='path', required=True, help='文件夹路径')
    parser.add_argument('--c', '--category', dest='category', 
                       choices=['landscape', 'color', 'animal'], required=True, help='分类维度')
    parser.add_argument('--m', '--min_count', dest='min_count', type=int, default=20, help='最小分类图片数量')
    parser.add_argument('--n', '--clusters_count', dest='clusters_count', type=int, default=10, help='聚类数上限，非全组合时为分组数量')
    parser.add_argument('--t', '--tol', dest='tol', type=float, default=1e-4, help='收敛阈值, 越小越精确，最大不超过0.1，默认1e-4，只在非全组合时有效')
    parser.add_argument('--f', '--full_combination', dest='full_combination', action='store_true', help='是否使用全组合')
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
    
    # 分类标准：使用KMeans聚类
    # n_clusters是类别数量，取值范围[1, l图片总数]，
    # tol是收敛阈值，控制算法停止的精度,越小越精确，
    #   tol 默认为1e-4(0.0001)， 理论可为任意小的正浮点数（如1e-6），最大不超过0.1, 
    #optimal_clusters = max(1, min(10, len(features_array)//args.min_count))
    optimal_clusters = max(2, args.clusters_count)  # 确保至少2个聚类
    
    if args.full_combination:        
        # 初始化参数范围
        n_clusters_range = range(2, optimal_clusters + 1)  # 假设聚类数从2到optimal_clusters
        tol_values = [1e-6]  # 初始tol值
        
        # 生成tol值序列，直到超过0.1
        while tol_values[-1] < 0.1:
            last_tol = tol_values[-1]
            next_tol = last_tol * 10
            if next_tol <= 0.1:
                tol_values.append(next_tol)
            else:
                tol_values.append(0.1)
        
        # 遍历所有组合
        for n_clusters_val in n_clusters_range:
            for tol_val in tol_values:
                print(f"--------")
                clusters = get_clusters(features_array, n_clusters=n_clusters_val, tol=tol_val)
                print(f"n_clusters={n_clusters_val}, tol={tol_val}")
                cluster_counts = Counter(clusters)
                export_clusters_to_excel(valid_paths, clusters, args.path, category_val=args.category, n_clusters_val=n_clusters_val, tol_val=tol_val)
                # 按数量降序排列并打印
                sorted_counts = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
                print("Category    File")
                for cluster, count in sorted_counts:
                    print(f"{cluster}    {count}")
    else:
        clusters = get_clusters(features_array, n_clusters=optimal_clusters, tol=args.tol)
        print(f"n_clusters={optimal_clusters}, tol={args.tol}")
        export_clusters_to_excel(valid_paths, clusters, args.path, category_val=args.category, n_clusters_val=optimal_clusters, tol_val=args.tol)
        for i, (path, cluster) in enumerate(zip(valid_paths, clusters)):
            print(f"{cluster}    {os.path.basename(path)}")
        cluster_counts = Counter(clusters)
        
        # 按数量降序排列并打印
        sorted_counts = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        print("Category    File")
        for cluster, count in sorted_counts:
            print(f"{cluster}    {count}")

if __name__ == '__main__':
    main()
