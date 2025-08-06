
from transformers import pipeline
import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_classifier():
    # 使用轻量级模型适配CPU环境
    return pipeline("zero-shot-classification", 
                  model="facebook/bart-large-mnli",
                  device=-1)  # 强制使用CPU

def classify_text(text, classifier, hierarchy):
    # 确保一级标签为字符串
    l1_labels = list(hierarchy.keys())
    l1_result = classifier(text, candidate_labels=l1_labels)
    l1 = str(l1_result['labels'])  # 强制转换首个标签为字符串
    
    # 二级分类处理
    try:
        l2_labels = hierarchy[l1]  # 直接使用字符串键访问
    except (KeyError, TypeError):
        l2_labels = []
    
    if l2_labels:
        l2_result = classifier(text, candidate_labels=l2_labels)
        l2 = str(l2_result['labels']) if l2_result['scores'] > 0.5 else "(Other)"
    else:
        l2 = "(N/A)"
    
    return l1, l2


def enforce_hierarchy(text, classifier, hierarchy):
    # 一级分类：取最高置信度标签
    l1_candidates = list(hierarchy.keys())
    l1_results = classifier(text, candidate_labels=l1_candidates)
    print(f"一级分类结果: {l1_results}")
    max_l1_score = max(l1_results['scores'])
    l1 = str(l1_results['labels'][l1_results['scores'].index(max_l1_score)])  # 强制选择首位标签
    if not l1:  # 如果一级分类结果为空
        return "(Other)", "(N/A)"
    
    # 二级分类处理
    l2 = "(N/A)"
    if l1 in hierarchy:
        sub_labels = list(hierarchy[l1])
        if sub_labels:  # 存在子类才处理
            l2_results = classifier(text, candidate_labels=sub_labels)
            max_l2_score = max(l2_results['scores'])
            if max_l2_score > 0.4:  # 置信度阈值
                l2 = str(l2_results['labels'][l2_results['scores'].index(max_l2_score)])

    return l1, l2


def process_file(input_file, output_file, hierarchy):
    classifier = load_classifier()
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            l1, l2 = enforce_hierarchy(line, classifier, hierarchy)
            print(f"Line {line_num}        {l1}        {l2}")
            fout.write(f"{line_num}        {l1}        {l2}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python feedback_classifier.py <input.txt> <output.txt>")
        sys.exit(1)
    
    # 用户提供的分类体系
    category_hierarchy =  {
        "AI-generated image issues": [
            "Complaints about AI images being unrealistic/poor quality",
            "Requests for an option to turn off AI images"
        ],
        "Insufficient support for multiple monitors": [
            "Cannot set different wallpapers for different screens",
            "Animated wallpapers only work on the main screen"
        ],
        "Wallpaper auto-change failure": [
            "Daily refresh not working",
            "Automatically overwritten after manual selection"
        ],
        "Image information missing": [
            "Missing location/photographer information",
            "Wrong geotags"
        ],
        "Political/cultural content controversy": [
            "Oppose LGBTQ+ wallpapers",
            "Oppose national flag wallpapers"
        ],
        "Image display anomalies": [
            "Images are cropped/stretched",
            "Resolution issues"
        ],
        "Animal image aversion": [
            "Fear of specific animals",
            "Requests to exclude animal categories"
        ],
        "Feature request: Theme classification": [
            "Need more theme options"
        ],
        "Browser forced jump": [
            "Clicking the desktop automatically opens the browser"
        ],
        "Excessive resource usage": [
            "Memory/CPU usage issues"
        ],
        "Installation/uninstallation issues": [
            "Installation without consent",
            "Difficult to uninstall"
        ],
        "Images appear repeatedly": [
            "Same images appear repeatedly"
        ],
        "Feature request: Collection feature": [
            "Request to collect favorite wallpapers"
        ],
        "Regional preferences": [
            "Request more localized content"
        ],
        "Poor image quality": [
            "Blurry/pixelation"
        ],
        "Feature request: Change frequency": [
            "Want to customize the change interval"
        ],
        "Widget issues": [
            "Widget location/function issues"
        ],
        "Inaccurate content": [
            "Images do not match descriptions"
        ],
        "Feature request: Dark mode": [
            "Need dark theme wallpapers"
        ],
        "Ads interfere": [
            "Hate promotional content"
        ],
        "Feature request: Rating system": [
            "Want like/dislike features"
        ],
        "System compatibility issues": [
            "Specific Windows version issues"
        ],
        "Slow update of image gallery": [
            "Theme images do not update"
        ],
        "Feature request: Custom images": [
            "Want to add personal images"
        ],
        "Positive feedback": [
            "Compliments on wallpapers"
        ],
        "Negative emotions (no specific content)": [
            "Simply express dislike"
        ],
        "Feature request: Exclude specific content": [
            "Want to block specific types of images"
        ],
        "Other technical issues": [
            "Various uncertain technical issues"
        ]
    }

    process_file(sys.argv[1], sys.argv[2], category_hierarchy)
