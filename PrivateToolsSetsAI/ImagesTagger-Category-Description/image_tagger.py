import os
import json
import torch
from PIL import Image
import clip
from tqdm import tqdm

LABEL_CATEGORIES = {
    "animals": ["cat", "dog", "lion", "bird", "fish", "elephant", "lynx", "leopard", "monkey", "deer", "dolphin", "zebra", "frog", "tiger", "bear", "weasel", "panda", "fox", "horse", "kangaroo", "orangutan", "squirrel", "duck", "rabbit", "wolf", "hedgehog", "eagle", "lizard", "beaver", "hyena", "owl", "goose", "antelope", "badger", "snake", "mouse", "seal", "turtle", "tropical fish", "flamingo", "swan", "peacock", "parrot", "whale", "raccoon", "moose", "camel", "donkey"],
    "terrain": ["cliff", "mountain", "lake", "seaside", "beach", "city", "town", "hills", "forest", "desert", "river", "valley", "canyon", "island", "jungle", "meadow", "ocean", "prairie", "swamp", "volcano", "waterfall", "glacier", "cave", "field", "garden", "park", "pond", "bay", "gorge", "marsh", "tundra", "savanna", "archipelago", "reef", "wasteland", "steppe", "plateau", "frozen lake", "hot spring", "spring", "oasis", "cove", "gulf", "delta", "fjord", "estuary", "bayou", "coral reef", "tide pool", "cliffside", "rocky shore", "sand dune", "iceberg", "canyon", "badlands", "badlands", "highland", "lowland", "space"],
    "weather": ["sunny", "rainy", "cloudy", "snowy", "foggy", "stormy", "windy", "hazy", "clear", "overcast"],
    "mood": ["peaceful", "gloomy", "serene", "agitated", "romantic"],
    "plants": ["tree", "flower", "grass", "bamboo", "cactus"],
    "scenery": ["wave", "vortex", "rock", "whirlwind", "boat", "car", "table", "chair", "building", "bridge", "sky", "cloud", "sunset", "sunrise", "night", "galaxy",  "daytime", "dawn", "dusk", "rainbow", "reflection", "shadow", "light", "darkness", "glow", "sparkle", "shine", "glimmer", "twilight", "moonlight", "starlight", "sunbeam", "mist", "dew", "frost", "blizzard", "thunderstorm", "hurricane", "tornado", "cyclone", "earthquake", "avalanche", "landslide", "tsunami", "volcanic eruption", "meteor shower", "solar flare", "aurora", "comet", "eclipse", "lightning", "rainstorm", "hailstorm", "snowstorm", "windstorm", "dust storm", "sandstorm", "flood", "drought", "heatwave", "cold snap", "frostbite", "glacier melt", "ice storm", "thunderclap", "whirlwind", "waterspout", "firestorm", "starfall", "lightning strike", "rainbow", "sun halo", "moon halo", "fog bank", "misty morning", "clear night", "starry sky", "cloudy sky", "blue sky", "sunny day", "rainy day", "snowy day", "windy day", "moon", "sun", "earth", "stars", "galaxy", "universe", "nebula", "comet", "asteroid", "black hole", "supernova", "quasar", "pulsar", "dark matter", "dark energy", "cosmic rays", "solar system", "exoplanet"]
}


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def generate_tags(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") 
                           for category in LABEL_CATEGORIES.values() 
                           for label in category]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    top_values, top_indices = similarity.topk(5)
    all_labels = [label for category in LABEL_CATEGORIES.values() for label in category]
    # 添加解包操作
    batch_indices = [idx for batch in top_indices for idx in batch]
    return [all_labels[int(i)] for i in batch_indices]


def generate_tags2(image_path, model, preprocess, device, 
                 confidence_thresh=0, max_tags=2):
    # 加载并预处理图像
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 生成所有标签的文本输入
    text_inputs = torch.cat([
        clip.tokenize(f"a photo of a {label}") 
        for category in LABEL_CATEGORIES.values() 
        for label in category
    ]).to(device)
    
    # 计算相似度
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # 按分类处理标签
    result_tags = []
    label_list = [label for cat in LABEL_CATEGORIES.values() for label in cat]
    
    for cat_name, cat_labels in LABEL_CATEGORIES.items():
        # 获取当前分类的索引范围
        cat_indices = torch.tensor([i for i, label in enumerate(label_list) 
                                  if label in cat_labels]).to(device)
        cat_similarity = similarity[0, cat_indices]
        
        # 应用置信度阈值
        valid_mask = cat_similarity > confidence_thresh
        valid_indices = torch.nonzero(valid_mask).flatten()
        
        if len(valid_indices) == 0:
            continue
            
        # 按相似度排序
        sorted_values, sorted_indices = cat_similarity[valid_indices].sort(descending=True)
        print(f"Processing category: {cat_name}, valid indices: {valid_indices}, sorted indices: {sorted_indices}")
        
        # 特殊处理scenery分类
        max_select = max_tags + 3 if cat_name == 'scenery' else max_tags
        selected_indices = [int(i) for i in valid_indices[sorted_indices[:max_select]]]
        
        # 添加有效标签
        for idx in selected_indices:  # 确保idx在有效范围内
            for i, cat_idx in enumerate(cat_indices):  # 遍历每个索引
                adjusted_idx = idx - int(cat_idx)
                if 0 <= adjusted_idx < len(cat_labels):
                    result_tags.append((cat_labels[adjusted_idx], cat_similarity[i].item()))
    
    print(result_tags)
    return sorted(result_tags, key=lambda x: -x[1])  # 按置信度降序


def process_folder(folder_path, output_file="tags.json"):
    model, preprocess, device = load_model()
    results = []
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            print(f"Processing image: {img_path}")
            tags = generate_tags(img_path, model, preprocess, device)
            results.append({"img": filename, "tags": tags})
            print(f"Generated tags for {filename}: {tags}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    process_folder("D:\\GrowthAI\\4 ReGroupInDimensions\\Total")
