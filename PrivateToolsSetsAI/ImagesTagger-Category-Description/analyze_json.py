import json
from collections import Counter, defaultdict

# 替换成你的 JSON 文件路径
json_file_path = "Total_full.json"

# 加载 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化计数器
country_counter = Counter()
region_counter = Counter()
category_counter = Counter()
category_labels = defaultdict(Counter)

# 遍历每一项
for item in data:
    location = item.get("location", {})
    country = location.get("country", "Unknown")
    region = location.get("region", "Unknown")
    country_counter[country] += 1
    region_counter[region] += 1

    tags = item.get("tags", [])
    for tag in tags:
        category = tag.get("category", "uncategorized")
        label = tag.get("label", "unknown")
        category_counter[category] += 1
        category_labels[category][label] += 1

# 打印统计结果
def print_counter(title, counter):
    print(f"\n=== {title} ===")
    for key, count in counter.most_common():
        print(f"{key}: {count}")

print_counter("Countries", country_counter)
print_counter("Regions", region_counter)
print_counter("Categories", category_counter)

print("\n=== Labels per Category ===")
for category, labels in category_labels.items():
    print(f"\n-- Category: {category} --")
    for label, count in labels.most_common():
        print(f"{label}: {count}")
