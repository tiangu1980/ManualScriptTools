import os
import json
import pandas as pd
import torch
import clip
from PIL import Image
from collections import defaultdict, Counter
from tqdm import tqdm

# ---------- 标签设置 ----------
LABEL_CATEGORIES = {
    "animals": ["cat", "dog", "lion", "bird", "fish", "elephant", "rabbit", "fox", "bear", "squirrel"],
    "terrain": ["mountain", "lake", "beach", "forest", "desert", "city", "ocean"],
    "weather": ["sunny", "rainy", "cloudy", "snowy", "stormy"],
    "mood": ["peaceful", "gloomy", "serene", "romantic"],
    "plants": ["tree", "flower", "grass", "bamboo", "cactus"],
    "scenery": ["sky", "cloud", "sunset", "night", "bridge", "moon", "sun", "star", "galaxy"]
}
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "an image of {}",
    "a scenic view of {}",
    "a beautiful {}",
]
MAX_TAGS_PER_CATEGORY = 3
MAX_TAGS_TOTAL = 7

# ---------- 特征构建 ----------
def build_text_features(model, device):
    label_info = []
    prompt_texts = []

    for category, labels in LABEL_CATEGORIES.items():
        for label in labels:
            for template in PROMPT_TEMPLATES:
                prompt = template.format(label)
                prompt_texts.append(prompt)
                label_info.append((label, category, prompt))

    text_inputs = torch.cat([clip.tokenize(p) for p in prompt_texts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features, label_info

# ---------- 单张图标签提取 ----------
def generate_tags(image_path, model, preprocess, device, text_features, label_info):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return []

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)

    prompt_per_label = len(PROMPT_TEMPLATES)
    label_scores = defaultdict(list)

    for i in range(0, len(label_info), prompt_per_label):
        label, category = label_info[i][0], label_info[i][1]
        scores = similarity[i:i + prompt_per_label]
        avg_score = scores.mean().item()
        if avg_score > 0:
            label_scores[label].append(avg_score)

    # 计算每个标签平均分
    tag_scores = {k: sum(v) / len(v) for k, v in label_scores.items()}
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    top_tags = [label for label, score in sorted_tags[:MAX_TAGS_TOTAL]]
    return top_tags

# ---------- 主处理逻辑 ----------
def process_excel_and_generate_keywords(excel_path):
    base_dir = os.path.dirname(excel_path)
    df = pd.read_excel(excel_path)
    grouped = df.groupby("Category")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_features, label_info = build_text_features(model, device)

    output_rows = []
    used_tag_sets = set()

    for category, group in tqdm(grouped, desc="Processing categories"):
        all_tags = []
        for _, row in group.iterrows():
            image_path = os.path.join(base_dir, row["File"])
            tags = generate_tags(image_path, model, preprocess, device, text_features, label_info)
            all_tags.extend(tags)

        tag_counts = Counter(all_tags)
        top_tags = [t for t, _ in tag_counts.most_common(MAX_TAGS_TOTAL)]

        # 防止标签组完全重复
        while tuple(top_tags) in used_tag_sets and len(tag_counts) > len(top_tags):
            for tag in tag_counts:
                if tag not in top_tags:
                    top_tags[-1] = tag
                    break

        used_tag_sets.add(tuple(top_tags))
        row = {"Category": category}
        for i, tag in enumerate(top_tags):
            row[f"Keyword{i+1}"] = tag
        output_rows.append(row)

    output_df = pd.DataFrame(output_rows)
    output_file = os.path.splitext(excel_path)[0] + "_tagged.xlsx"
    output_df.to_excel(output_file, index=False)
    print(f"\n✅ 处理完成！结果已保存为：{output_file}")

# ---------- 启动 ----------
if __name__ == "__main__":
    input_excel = "YOUR_EXCEL_FILE.xlsx"  # 👈 替换成你实际的文件名
    process_excel_and_generate_keywords(input_excel)
