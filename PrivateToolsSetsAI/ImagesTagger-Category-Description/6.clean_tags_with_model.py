import json
import argparse
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# 加载本地语义模型（只加载一次）
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def tags_conflict(tag1, tag2, threshold=0.3):
    sim = compute_similarity(tag1, tag2)
    return sim < threshold

def filter_animals(tags):
    top2 = sorted(tags, key=lambda x: -x["score"])[:2]
    for tag in top2:
        if tag["category"] == "animal":
            return [tag]
    return []

def filter_terrain(tags):
    terrain_tags = [t for t in tags if t["category"] == "terrain"]
    return sorted(terrain_tags, key=lambda x: -x["score"])[:2]

def filter_conflicts(tags):
    keep = []
    kinds = ["weather", "mood", "scenery"]
    grouped = defaultdict(list)
    for tag in tags:
        if tag["category"] in kinds:
            grouped[tag["category"]].append(tag)

    for kind, tlist in grouped.items():
        retained = []
        for tag in tlist:
            if not any(tags_conflict(tag["label"], other["label"]) for other in retained):
                retained.append(tag)
        if not retained and tlist:
            retained = [sorted(tlist, key=lambda x: -x["score"])[0]]
        keep.extend(retained)
    return keep

def filter_plants(tags, context_labels, threshold=0.35):
    plant_tags = [t for t in tags if t["category"] == "plants"]
    keep = []
    for tag in plant_tags:
        label = tag["label"]
        scores = [compute_similarity(label, ctx) for ctx in context_labels]
        if scores and max(scores) >= threshold:
            keep.append(tag)
    return keep

def clean_image_tags(image_obj):
    tags = image_obj["tags"]
    others = [t for t in tags if t["category"] not in {"animal", "terrain", "weather", "mood", "scenery", "plants"}]

    animal_filtered = filter_animals(tags)
    terrain_filtered = filter_terrain(tags)
    conflict_filtered = filter_conflicts(tags)

    context_tags = animal_filtered + terrain_filtered + conflict_filtered + others
    context_labels = [t["label"] for t in context_tags]

    plant_filtered = filter_plants(tags, context_labels)

    final_tags = animal_filtered + terrain_filtered + conflict_filtered + plant_filtered + others
    final_tags = sorted(final_tags, key=lambda x: -x["score"])
    image_obj["tags"] = final_tags
    return image_obj

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for image_obj in data:
        cleaned.append(clean_image_tags(image_obj))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"✅ 清洗完成，输出到: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始标签 JSON 文件路径")
    parser.add_argument("--output", required=True, help="清洗后 JSON 文件路径")
    args = parser.parse_args()

    process_file(args.input, args.output)

if __name__ == "__main__":
    main()
