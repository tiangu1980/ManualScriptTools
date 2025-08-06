import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

# 定义标签分类
LABEL_CATEGORIES = {
    # 生物
    "animals": [
        "cat", "dog", "lion", "bird", "fish", "elephant", "lynx", "leopard", "monkey", "deer",
        "dolphin", "zebra", "frog", "tiger", "bear", "weasel", "panda", "fox", "horse", "kangaroo",
        "orangutan", "squirrel", "duck", "rabbit", "wolf", "hedgehog", "eagle", "lizard", "beaver",
        "hyena", "owl", "goose", "antelope", "badger", "snake", "mouse", "seal", "turtle",
        "tropical fish", "flamingo", "swan", "peacock", "parrot", "whale", "raccoon", "moose",
        "camel", "donkey"
    ],
    # 地形
    "terrain": [
        "cliff", "mountain", "lake", "seaside", "beach", "city", "town", "hills", "forest", "desert",
        "river", "valley", "canyon", "island", "jungle", "meadow", "ocean", "prairie", "swamp",
        "volcano", "waterfall", "glacier", "cave", "field", "garden", "park", "pond", "bay", "gorge",
        "marsh", "tundra", "savanna", "archipelago", "reef", "wasteland", "steppe", "plateau",
        "frozen lake", "hot spring", "spring", "oasis", "cove", "gulf", "delta", "fjord", "estuary",
        "bayou", "coral reef", "tide pool", "cliffside", "rocky shore", "sand dune", "iceberg",
        "highland", "lowland", "space"
    ],
    # 天气
    "weather": [
        "sunny", "rainy", "cloudy", "snowy", "foggy", "stormy", "windy", "hazy", "clear", "overcast"
    ],
    # 情绪
    "mood": [
        "peaceful", "gloomy", "serene", "agitated", "romantic"
    ],
    # 植物
    "plants": [
        "tree", "flower", "grass", "bamboo", "cactus"
    ],
    # 风景
    "scenery": [
        "wave", "vortex", "rock", "whirlwind", "boat", "car", "table", "chair", "building", "bridge",
        "sky", "cloud", "sunset", "sunrise", "night", "galaxy", "daytime", "dawn", "dusk", "rainbow",
        "reflection", "shadow", "light", "darkness", "glow", "sparkle", "shine", "glimmer", "twilight",
        "moonlight", "starlight", "sunbeam", "mist", "dew", "frost", "blizzard", "thunderstorm",
        "hurricane", "tornado", "cyclone", "earthquake", "avalanche", "landslide", "tsunami",
        "volcanic eruption", "meteor shower", "solar flare", "aurora", "comet", "eclipse", "lightning",
        "rainstorm", "hailstorm", "snowstorm", "windstorm", "dust storm", "sandstorm", "flood",
        "drought", "heatwave", "cold snap", "frostbite", "glacier melt", "ice storm", "thunderclap",
        "waterspout", "firestorm", "starfall", "lightning strike", "sun halo", "moon halo", "fog bank",
        "misty morning", "clear night", "starry sky", "cloudy sky", "blue sky", "sunny day",
        "rainy day", "snowy day", "windy day", "moon", "sun", "earth", "stars", "universe", "nebula",
        "asteroid", "black hole", "supernova", "quasar", "pulsar", "dark matter", "dark energy",
        "cosmic rays", "solar system", "exoplanet"
    ]
}

# Prompt 模板
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "an image showing {}",
    "a scene containing {}",
    "a picture with {} object"
]

MAX_TAGS_TOTAL = 15
MAX_TAGS_PER_CATEGORY = 3

def build_text_features(model, device):
    label_info = []  # [(label, category, prompt)]
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

def generate_tags(image_path, model, preprocess, device, text_features, label_info):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return []

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)  # shape: [N prompts]

    # 聚合每个标签的多个 prompt 分数
    category_scores = {}  # {category: [ (label, score) ]}
    prompt_per_label = len(PROMPT_TEMPLATES)
    seen_labels = set()

    for i in range(0, len(label_info), prompt_per_label):
        prompts = label_info[i:i + prompt_per_label]
        label, category = prompts[0][0], prompts[0][1]
        if (label, category) in seen_labels:
            continue
        seen_labels.add((label, category))

        scores = similarity[i:i + prompt_per_label]
        avg_score = scores.mean().item()
        if avg_score > 0:
            category_scores.setdefault(category, []).append({
                "label": label,
                "category": category,
                "score": round(avg_score, 4)
            })

    # 类内取前 N 个
    selected_tags = []
    for cat, items in category_scores.items():
        
        if cat == "animals":
            top_cat_tags = sorted(items, key=lambda x: x["score"], reverse=True)[:2]
        else:
            top_cat_tags = sorted(items, key=lambda x: x["score"], reverse=True)[:MAX_TAGS_PER_CATEGORY]
        selected_tags.extend(top_cat_tags)

    # 全局排序并限制总数量
    selected_tags = sorted(selected_tags, key=lambda x: x["score"], reverse=True)[:MAX_TAGS_TOTAL]
    return selected_tags

def process_folder(folder_path, model, preprocess, device, text_features, label_info):
    result = []
    for filename in tqdm(os.listdir(folder_path)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        full_path = os.path.join(folder_path, filename)
        tags = generate_tags(full_path, model, preprocess, device, text_features, label_info)
        result.append({"image": filename, "tags": tags})
    return result

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 一次性构建文本向量
    text_features, label_info = build_text_features(model, device)

    image_folder = "D:\\GrowthAI\\4 ReGroupInDimensions\\Total"  # 👈 替换为你的图片目录
    output_path = "Total.json"

    results = process_folder(image_folder, model, preprocess, device, text_features, label_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 所有图片处理完成，标签已保存到：{output_path}")

if __name__ == "__main__":
    main()
