import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_all_images_from_paths(paths):
    image_set = set()
    for folder in paths:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    image_set.add(Path(file).name)
    return image_set

def analyze(json_data, image_paths_set, target_category=None, target_label=None):
    total_images = len(json_data)
    match_topcat_images = []
    
    # 1. 统计 json 中 tags[0] 为目标 category/label 的图片
    for item in json_data:
        tags = item.get("tags", [])
        if not tags:
            continue
        top_tag = tags[0]
        if (target_category and top_tag["category"] == target_category) or \
           (target_label and top_tag["label"] == target_label):
            match_topcat_images.append(item["image"])

    match_topcat_set = set(match_topcat_images)
    found_in_paths = match_topcat_set.intersection(image_paths_set)

    # 输出1
    print(f"1️⃣ JSON 中首标签为指定 {'category' if target_category else 'label'} 的图片数量: {len(match_topcat_images)}")
    if match_topcat_images:
        print(f"   其中在指定路径中找到的图片数量: {len(found_in_paths)}")
        print(f"   占比: {100 * len(found_in_paths) / len(match_topcat_images):.2f}%")
    else:
        print("   没有任何图片匹配")

    # 2. 路径中图片中，首标签为指定类型的图片统计
    in_path_with_topcat = 0
    for item in json_data:
        if item["image"] not in image_paths_set:
            continue
        tags = item.get("tags", [])
        if not tags:
            continue
        top_tag = tags[0]
        if (target_category and top_tag["category"] == target_category) or \
           (target_label and top_tag["label"] == target_label):
            in_path_with_topcat += 1

    # 输出2
    print(f"2️⃣ 指定路径中的图片数量: {len(image_paths_set)}")
    print(f"   其中首标签为指定类型的图片数量: {in_path_with_topcat}")
    if image_paths_set:
        print(f"   占比: {100 * in_path_with_topcat / len(image_paths_set):.2f}%")
    else:
        print("   路径中无有效图片")

def main():
    parser = argparse.ArgumentParser(description="Analyze tag categories or labels in JSON and image paths")
    parser.add_argument("--json", default="tags.json", help="Path to the JSON file")
    parser.add_argument("--category", help="Target tag category")
    parser.add_argument("--label", help="Target tag label")
    parser.add_argument("--pathes", nargs="+", required=True, help="List of image folder paths")

    args = parser.parse_args()

    if not args.category and not args.label:
        print("Error: Must specify either --category or --label")
        return

    json_data = load_json_data(args.json)
    image_paths_set = collect_all_images_from_paths(args.pathes)
    analyze(json_data, image_paths_set, target_category=args.category, target_label=args.label)

if __name__ == "__main__":
    main()
