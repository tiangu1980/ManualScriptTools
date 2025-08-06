
import json
from collections import defaultdict

def analyze_json_data(file_path):
    # 初始化统计字典
    country_stats = defaultdict(int)
    region_stats = defaultdict(int)
    category_stats = defaultdict(int)
    label_by_category = defaultdict(lambda: defaultdict(int))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for item in data:
                # 统计country和region
                country = item['location']['country']
                region = item['location']['region']
                country_stats[country] += 1
                region_stats[region] += 1
                
                # 统计tags中的category和label
                for tag in item['tags']:
                    category = tag['category']
                    label = tag['label']
                    category_stats[category] += 1
                    label_by_category[category][label] += 1
                    
        # 打印统计结果
        print("\n=== 国家统计 ===")
        for country, count in sorted(country_stats.items()):
            print(f"{country}: {count}次")
            
        print("\n=== 地区统计 ===")
        for region, count in sorted(region_stats.items()):
            print(f"{region}: {count}次")
            
        print("\n=== 分类统计 ===")
        for category, count in sorted(category_stats.items()):
            print(f"{category}: {count}次")
            
        print("\n=== 每个分类下的标签统计 ===")
        for category, labels in sorted(label_by_category.items()):
            print(f"\n分类: {category}")
            for label, count in sorted(labels.items()):
                print(f"  {label}: {count}次")
                
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
    except json.JSONDecodeError:
        print("错误: 文件内容不是有效的JSON格式")
    except KeyError as e:
        print(f"错误: JSON结构不符合预期，缺少键: {e}")

if __name__ == "__main__":
    file_path = input("请输入JSON文件路径: ")
    analyze_json_data(file_path)
