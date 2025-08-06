import json
import argparse

def retain_top_category_tag(data, category):
    for image_data in data:
        tags = image_data.get("tags", [])

        # 拆成目标类别 和 其它类别
        category_tags = [tag for tag in tags if tag["category"] == category]
        other_tags = [tag for tag in tags if tag["category"] != category]

        # 如果目标类别存在，取分数最高的一个
        if category_tags:
            top_tag = max(category_tags, key=lambda x: x["score"])
            image_data["tags"] = other_tags + [top_tag]
        else:
            image_data["tags"] = tags  # 不变

    return data

def main():
    parser = argparse.ArgumentParser(description="Retain only the highest scoring tag for a specific category.")
    parser.add_argument("--input", required=True, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--category", required=True, help="Category to filter (e.g., 'weather').")
    args = parser.parse_args()

    # 加载数据
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据
    updated_data = retain_top_category_tag(data, args.category)

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
