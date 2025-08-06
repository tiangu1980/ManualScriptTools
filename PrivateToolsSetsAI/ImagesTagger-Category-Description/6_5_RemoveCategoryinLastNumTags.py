import json
import argparse

def remove_lowest_scoring_tags_with_category(data, category, remove_count):
    for image_data in data:
        tags = image_data.get("tags", [])
        # 取分数最低的 N 个标签
        lowest_tags = sorted(tags, key=lambda x: x["score"])[:remove_count]

        # 找出这些最低分的标签中，属于指定 category 的
        to_remove_set = set(id(tag) for tag in lowest_tags if tag["category"] == category)

        # 重新保留 tags：排除需要移除的
        image_data["tags"] = [tag for tag in tags if id(tag) not in to_remove_set]
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Remove lowest scoring tags from JSON if they belong to specified category.")
    parser.add_argument("--input", required=True, help="Input JSON file path.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument("--category", required=True, help="Category to remove (e.g., 'weather').")
    parser.add_argument("--remove-count", type=int, required=True, help="Number of lowest scoring tags to consider per image.")
    args = parser.parse_args()

    # 加载输入文件
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 执行处理
    updated_data = remove_lowest_scoring_tags_with_category(data, args.category, args.remove_count)

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
