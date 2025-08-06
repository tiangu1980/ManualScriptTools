import argparse
import json
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sentences(data, top_k=15):
    sentences = []
    filenames = []
    for item in data:
        tags = item.get("tags", [])
        sorted_tags = sorted(tags, key=lambda x: -x["score"])
        top_tags = [t["label"] for t in sorted_tags[:top_k]]
        sentence = " ".join(top_tags)
        sentences.append(sentence)
        filenames.append(item["image"])
    return sentences, filenames


def generate_title_and_description(tag_lists):
    all_tags = [tag for tags in tag_lists for tag in tags]
    counter = Counter(all_tags)
    top_tags = [tag for tag, _ in counter.most_common(4)]
    title = " / ".join(top_tags).title()
    #description = f"Images featuring {', '.join(top_tags[:-1])}, and {top_tags[-1]}."
    description = " "
    return title, description


def cluster_single(data, embeddings, filenames, min_size, max_size, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    results = []
    for label, idxs in clusters.items():
        if not (min_size <= len(idxs) <= max_size):
            continue

        tag_lists = []
        files = []

        for idx in idxs:
            tags = data[idx].get("tags", [])
            sorted_tags = sorted(tags, key=lambda x: -x["score"])
            top_tags = [t["label"] for t in sorted_tags[:15]]
            tag_lists.append(top_tags)
            files.append(filenames[idx])

        title, description = generate_title_and_description(tag_lists)
        results.append({
            "theme": title,
            "description": description,
            "images": files
        })

    return results


def cluster_multi(data, embeddings, filenames, min_size, max_size, num_clusters, sim_threshold=0.7):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    centers = kmeans.cluster_centers_

    sim_matrix = cosine_similarity(embeddings, centers)

    # 图片可出现在多个主题中
    clusters = defaultdict(list)
    for img_idx, sim_row in enumerate(sim_matrix):
        for cluster_idx, sim_score in enumerate(sim_row):
            if sim_score >= sim_threshold:
                clusters[cluster_idx].append(img_idx)

    results = []
    for label, idxs in clusters.items():
        if not (min_size <= len(idxs) <= max_size):
            continue

        tag_lists = []
        files = []

        for idx in idxs:
            tags = data[idx].get("tags", [])
            sorted_tags = sorted(tags, key=lambda x: -x["score"])
            top_tags = [t["label"] for t in sorted_tags[:15]]
            tag_lists.append(top_tags)
            files.append(filenames[idx])

        title, description = generate_title_and_description(tag_lists)
        results.append({
            "theme": title,
            "description": description,
            "images": list(set(files))  # remove duplicates within one cluster
        })

    return results


# 加载本地模型路径
model_path_l6 = "D:\\GrowthAI\\4 ReGroupInDimensions\\all-MiniLM-L6-v2"
model_path_l12 = "D:\\GrowthAI\\4 ReGroupInDimensions\\paraphrase-multilingual-MiniLM-L12-v2"

def main(input_path, output_path, min_size, max_size, num_clusters, multi_theme):
    data = load_json(input_path)
    sentences, filenames = generate_sentences(data)

    print(f"Generating embeddings for {len(sentences)} images...")
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    #model = SentenceTransformer(model_path_l6)  # 使用本地模型
    model = SentenceTransformer(model_path_l12)  # 使用本地模型
    embeddings = model.encode(sentences)

    print(f"Clustering with {'multi-theme' if multi_theme else 'single-theme'} mode...")
    if multi_theme:
        results = cluster_multi(data, embeddings, filenames, min_size, max_size, num_clusters)
    else:
        results = cluster_single(data, embeddings, filenames, min_size, max_size, num_clusters)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} themes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster images into themed groups with descriptions")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument("--min-size", type=int, default=20, help="Minimum images per theme")
    parser.add_argument("--max-size", type=int, default=60, help="Maximum images per theme")
    parser.add_argument("--clusters", type=int, default=30, help="Target number of clusters")
    parser.add_argument("--multi-theme", action="store_true", help="Allow images to appear in multiple themes")

    args = parser.parse_args()
    main(args.input, args.output, args.min_size, args.max_size, args.clusters, args.multi_theme)
