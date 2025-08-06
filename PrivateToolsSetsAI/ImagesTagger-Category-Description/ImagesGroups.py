import json
from collections import defaultdict, Counter

# 加载 JSON 文件
with open("Total_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 每个 category 的最大数量限制（最小值默认为 0）
category_limits = {
    "animals": 1,
    "terrain": 3,
    "weather": 2,
    "mood": 2,
    "plants": 2,
    "scenery": 3,
}

target_categories = list(category_limits.keys())
combination_counter = Counter()

# 遍历每张图片
for item in data:
    country = item.get("location", {}).get("country", "Unknown")
    tags = item.get("tags", [])

    # 将标签按 category 分类
    category_to_labels = defaultdict(list)
    for tag in tags:
        cat = tag.get("category")
        label = tag.get("label")
        if cat in category_limits:
            category_to_labels[cat].append(label)

    # 保证每个 category 都有值（或者是 NONE）
    final_labels = {}

    for cat in target_categories:
        labels = category_to_labels.get(cat, [])
        max_count = category_limits[cat]

        if not labels:
            if cat == "animals":
                final_labels[cat] = ["NONE"]  # 单独处理 animals
            else:
                final_labels[cat] = ["NONE"]
        else:
            if cat == "animals":
                final_labels[cat] = [labels[0]]  # animals 只取第一个
            else:
                final_labels[cat] = labels[:max_count]

    # 构造组合字符串
    combo_str = (
        f'In country: "{country}" ; '
        f'terrain is "{", ".join(final_labels["terrain"])}" ; '
        f'animals "{final_labels["animals"][0]}" ; '
        f'plants is "{", ".join(final_labels["plants"])}" ; '
        f'weather is "{", ".join(final_labels["weather"])}" ; '
        f'mood is "{", ".join(final_labels["mood"])}" ; '
        f'scenery with "{", ".join(final_labels["scenery"])}"'
    )

    combination_counter[combo_str] += 1

# 输出统计结果
for combo, count in combination_counter.most_common():
    print(f"{combo} = {count}")
