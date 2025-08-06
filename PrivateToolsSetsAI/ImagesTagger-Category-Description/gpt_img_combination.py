import json
import itertools
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------- CLI 参数 ----------
parser = argparse.ArgumentParser(description='Generate image combinations based on location and tags.')
parser.add_argument('--input', required=True, help='Path to input JSON file.')
parser.add_argument('--output', required=True, help='Output directory.')
parser.add_argument('--min-threshold', type=int, default=5, help='Minimum image count threshold per attribute or combination.')
parser.add_argument('--threads', type=int, default=8, help='Number of worker threads to use.')
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_DIR = args.output
MIN_IMAGE_THRESHOLD = args.min_threshold
THREAD_COUNT = args.threads

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 分类规则 ----------
MAX_SCENERY = 2
MAX_WEATHER = 2
MAX_MOOD = 2
MAX_PLANTS = 2
MAX_TERRAIN = 2

category_rules = {
    'animals': (1, 1),
    'terrain': (1, MAX_TERRAIN),
    'weather': (1, MAX_WEATHER),
    'mood': (1, MAX_MOOD),
    'plants': (1, MAX_PLANTS),
    'scenery': (1, MAX_SCENERY),
}

# ---------- 全局变量 ----------
combo_descriptions = []
combo_lock = threading.Lock()
combo_id = 1

# ---------- 读取数据 ----------
print(f"[INFO] Loading data from {INPUT_FILE}...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"[INFO] Loaded {len(data)} images.")

# ---------- 索引数据 ----------
category_to_labels = defaultdict(lambda: defaultdict(set))  # category -> label -> set(images)
country_region_to_images = defaultdict(set)

for item in data:
    img = item['image']
    country = item['location'].get('country', 'No Info')
    region = item['location'].get('region', 'No Info')
    country_region_to_images[(country, region)].add(img)

    for tag in item['tags']:
        category = tag['category']
        label = tag['label']
        category_to_labels[category][label].add(img)

# ---------- 过滤掉标签出现频率低的 ----------
filtered_category_labels = defaultdict(list)
for category, label_map in category_to_labels.items():
    for label, images in label_map.items():
        if len(images) >= MIN_IMAGE_THRESHOLD:
            filtered_category_labels[category].append(label)
print(f"[INFO] Filtered categories: {len(filtered_category_labels)} with minimum threshold {MIN_IMAGE_THRESHOLD}")

# 有效分类
available_categories = [cat for cat in category_rules if filtered_category_labels[cat]]
print(f"[INFO] Available categories: {available_categories}")

# 有效 location（含 None 表示不限）
valid_locations = [loc for loc, imgs in country_region_to_images.items() if len(imgs) >= MIN_IMAGE_THRESHOLD]
valid_locations.append((None, None))
print(f"[INFO] Valid locations: {len(valid_locations)}")

# ---------- 分类组合 ----------
def generate_category_combos():
    results = []
    for n in range(1, len(available_categories) + 1):
        for cat_combo in itertools.combinations(available_categories, n):
            results.append(cat_combo)
    return results

category_combos = generate_category_combos()

# ---------- 多线程处理函数 ----------
def process_combo(loc: Tuple[str, str], category_set: Tuple[str]):
    global combo_id
    cat_label_options = []
    for cat in category_set:
        labels = filtered_category_labels[cat]
        if not labels:
            continue
        min_count, max_count = category_rules[cat]
        label_combos = []
        for i in range(min_count, min(len(labels), max_count) + 1):
            label_combos.extend(itertools.combinations(labels, i))
        cat_label_options.append((cat, label_combos))

    if not cat_label_options:
        return []

    all_combos = list(itertools.product(*[[(cat, list(lbl)) for lbl in lbls] for cat, lbls in cat_label_options]))
    results = []

    for combo in all_combos:
        condition = defaultdict(set)
        for cat, labels in combo:
            condition[cat].update(labels)

        matched_images = []
        for item in data:
            img = item['image']
            country = item['location'].get('country', 'No Info')
            region = item['location'].get('region', 'No Info')

            # 匹配 location
            if loc[0] and country != loc[0]:
                continue
            if loc[1] and region != loc[1]:
                continue

            # 匹配标签
            tags_by_cat = defaultdict(set)
            for tag in item['tags']:
                tags_by_cat[tag['category']].add(tag['label'])

            matched = True
            for cat, required_labels in condition.items():
                if cat not in tags_by_cat or not tags_by_cat[cat].intersection(required_labels):
                    matched = False
                    break

            if matched:
                matched_images.append(img)

        if len(matched_images) >= MIN_IMAGE_THRESHOLD:
            with combo_lock:
                cid = combo_id
                combo_id += 1

            parts = [f"{cid}"]
            if loc[0]:
                parts.append(f'In country: "{loc[0]}"')
            if loc[1]:
                parts.append(f'In region: "{loc[1]}"')
            for cat in category_set:
                keyword = "with" if cat == "scenery" else "is"
                label_str = ', '.join(condition[cat])
                parts.append(f'{cat} {keyword} "{label_str}"')
            parts.append(f"= {len(matched_images)}")
            desc_str = " ; ".join(parts)

            # 文件名安全处理
            safe_desc = desc_str.replace('"', '').replace(',', '').replace(':', '').replace(';', '').replace(' ', '_')
            filename = f"{cid}_{len(matched_images)}_{safe_desc[:100]}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                for img in matched_images:
                    f.write(img + '\n')

            results.append(desc_str)
    return results

# ---------- 多线程启动 ----------
print(f"[INFO] Starting combination generation with {THREAD_COUNT} threads...")
tasks = []
with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
    for loc in valid_locations:
        for cat_combo in category_combos:
            tasks.append(executor.submit(process_combo, loc, cat_combo))

    for future in as_completed(tasks):
        descs = future.result()
        with combo_lock:
            combo_descriptions.extend(descs)

# ---------- 输出汇总文件 ----------
summary_file = os.path.join(OUTPUT_DIR, "combination_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    for line in combo_descriptions:
        f.write(line + '\n')

print(f"[DONE] 有效组合总数: {len(combo_descriptions)}，结果已保存到: {OUTPUT_DIR}")
