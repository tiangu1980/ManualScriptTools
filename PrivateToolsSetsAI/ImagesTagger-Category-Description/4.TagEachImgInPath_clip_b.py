import os
import json
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# 定义标签分类
LABEL_CATEGORIES = {
    "animals": [
        "cat", "dog", "lion", "tiger", "leopard", "lynx", "panther", "cheetah",
        "wolf", "fox", "bear", "panda", "hyena", "weasel", "badger",
        "elephant", "rhinoceros", "hippopotamus", "giraffe", "camel", "donkey", "horse", "zebra", "moose", "deer", "antelope",
        "rabbit", "mouse", "rat", "squirrel", "hedgehog", "beaver", "kangaroo", "koala", "raccoon", "monkey", "orangutan", "gorilla",
        "bird", "eagle", "owl", "parrot", "peacock", "flamingo", "duck", "goose", "swan", "pigeon", "seagull",
        "fish", "tropical fish", "shark", "whale", "dolphin", "seal", "sea lion", "octopus", "jellyfish", "starfish",
        "frog", "lizard", "snake", "turtle", "crocodile", "alligator", "chameleon",
        "insect", "butterfly", "bee", "ant", "dragonfly", "beetle", "spider"
    ],
    "terrain": [
        "cliff", "mountain", "hill", "valley", "canyon", "plateau", "ridge", "peak",
        "lake", "pond", "river", "stream", "creek", "waterfall", "frozen lake",
        "ocean", "sea", "bay", "gulf", "lagoon", "cove", "delta", "estuary", "fjord",
        "desert", "dune", "oasis", "wasteland", "savanna", "tundra", "steppe",
        "forest", "jungle", "woods", "grove", "swamp", "marsh", "wetland",
        "meadow", "field", "plain", "prairie", "garden", "park",
        "island", "archipelago", "reef", "coral reef",
        "cave", "glacier", "iceberg", "volcano", "hot spring", "spring",
        "rocky shore", "cliffside", "highland", "lowland", "gorge",
        "space", "crater", "lunar surface"
    ],
    "weather": [
        "sunny", "partly cloudy", "cloudy", "overcast", "clear", "hazy", "foggy", "misty",
        "rainy", "drizzling", "thunderstorm", "stormy", "rainstorm", "showers",
        "snowy", "light snow", "heavy snow", "blizzard", "snowstorm",
        "windy", "breezy", "gale", "dust storm", "sandstorm",
        "humid", "dry", "hot", "cold", "frosty", "icy", "freezing",
        "smoky", "hazy sky", "storm clouds", "clear sky", "dark clouds"
    ],
    "mood": [
        "peaceful", "serene", "calm", "romantic", "dreamy",
        "gloomy", "melancholic", "lonely", "mysterious", "eerie",
        "agitated", "chaotic", "tense", "dramatic", "intense",
        "joyful", "cheerful", "hopeful", "playful", "vibrant",
        "cold", "warm", "dark", "bright", "majestic", "inspiring",
        "tranquil", "reflective", "nostalgic", "uplifting", "spiritual"
    ],
    "plants": [
        "tree", "flower", "grass", "bamboo", "cactus", "shrub", "bush", "vine", "moss",
        "fern", "palm", "pine", "willow", "lotus", "orchid", "ivy", "lily", "oak", "maple",
        "spruce", "sunflower", "dandelion", "rose", "tulip"
    ],
    "scenery": [
        "wave", "rock", "boat", "car", "building", "bridge", "sky", "cloud", "sunset", "sunrise",
        "night", "galaxy", "dawn", "dusk", "rainbow", "reflection", "shadow", "light", "darkness",
        "glow", "sparkle", "shine", "twilight", "moonlight", "starlight", "sunbeam", "mist", "dew",
        "frost", "thunderstorm", "hurricane", "tornado", "cyclone", "earthquake", "avalanche",
        "landslide", "tsunami", "volcanic eruption", "meteor shower", "solar flare", "aurora",
        "comet", "eclipse", "lightning", "rainstorm", "hailstorm", "snowstorm", "windstorm",
        "dust storm", "sandstorm", "flood", "drought", "heatwave", "cold snap", "ice storm",
        "waterspout", "firestorm", "starfall", "sun halo", "moon halo", "fog bank", "clear night",
        "starry sky", "cloudy sky", "blue sky", "sunny day", "rainy day", "snowy day", "windy day",
        "moon", "sun", "earth", "stars", "universe", "nebula", "asteroid", "black hole",
        "supernova", "quasar", "pulsar", "dark matter", "solar system", "exoplanet"
    ]
}

PROMPT_TEMPLATES_BY_CATEGORY = {
    "animals": [
        "a photo of a {}", "an animal: {}", "a picture showing a {}",
        "a wildlife photo of a {}", "a creature such as a {}"
    ],
    "terrain": [
        "a landscape with {}", "a photo of {} terrain", "a ground surface made of {}",
        "the landform is {}", "a terrain type: {}"
    ],
    "weather": [
        "the weather looks {}", "a {} day", "a scene with {} weather",
        "the sky is {}", "it is a {} weather condition"
    ],
    "mood": [
        "the image feels {}", "this gives a {} atmosphere", "the mood is {}",
        "a {} emotional tone", "this photo looks very {}"
    ],
    "plants": [
        "a photo of {} plants", "there are {} growing", "a picture of {} vegetation",
        "the plant type is {}", "a scene with {} flora"
    ],
    "scenery": [
        "a photo of {} scenery", "a beautiful view of {}", "the scene shows {}",
        "a {} natural landscape", "a {} travel destination"
    ]
}

MAX_TAGS_TOTAL = 15
MAX_TAGS_PER_CATEGORY = 3

def build_text_features(model, tokenizer, device):
    label_info = []
    prompt_texts = []

    for category, labels in LABEL_CATEGORIES.items():
        prompts = PROMPT_TEMPLATES_BY_CATEGORY.get(category, ["a photo of a {}"])
        for label in labels:
            for template in prompts:
                prompt = template.format(label)
                prompt_texts.append(prompt)
                label_info.append((label, category, prompt))

    text_tokens = tokenizer(prompt_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
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

        similarity = (image_features @ text_features.T).squeeze(0)

    category_scores = {}
    prompt_per_label = 5
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

    selected_tags = []
    for cat, items in category_scores.items():
        if cat == "animals":
            top_cat_tags = sorted(items, key=lambda x: x["score"], reverse=True)[:2]
        else:
            top_cat_tags = sorted(items, key=lambda x: x["score"], reverse=True)[:MAX_TAGS_PER_CATEGORY]
        selected_tags.extend(top_cat_tags)

    selected_tags = sorted(selected_tags, key=lambda x: x["score"], reverse=True)[:MAX_TAGS_TOTAL]
    return selected_tags

def process_folder(folder_path, model, preprocess, device, text_features, label_info):
    result = []
    for filename in tqdm(os.listdir(folder_path)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        full_path = os.path.join(folder_path, filename)
        tags = generate_tags(full_path, model, preprocess, device, text_features, label_info)
        result.append({"image": filename, "tags": tags})
    return result

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L-14"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    text_features, label_info = build_text_features(model, tokenizer, device)

    image_folder = "D:\\GrowthAI\\4 ReGroupInDimensions\\Total"
    output_path = "Total_b.json"

    results = process_folder(image_folder, model, preprocess, device, text_features, label_info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 所有图片处理完成，标签已保存到：{output_path}")

if __name__ == "__main__":
    main()
