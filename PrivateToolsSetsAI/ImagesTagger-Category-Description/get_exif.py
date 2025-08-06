import os
import json
import re
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_data(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            return {TAGS.get(tag, tag): value 
                   for tag, value in exif_data.items()}
    except (AttributeError, IOError, KeyError):
        return None
    return None


# 清理描述中的特殊字符
def clean_description(description: str) -> str:
    """
    替换 description 中的特殊字符为单个空格，保留英文、数字、空格和常见字母符号。
    """
    if not description:
        return ""

    # 用正则表达式替换所有非字母数字和空格的字符为一个空格
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", description)

    # 再将多个连续空格压缩成一个
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()

def print_exif_for_folder(folder_path):
    result = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            filepath = os.path.join(folder_path, filename)
            exif = get_exif_data(filepath)
            print(f"\n=== {filepath} ===")
            
            description_value = ""
            if exif:
                for tag_id, value in exif.items():
                    print(f"{tag_id}: {value}")
                    
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "ImageDescription":
                        # Convert bytes to string if necessary
                        if isinstance(value, bytes):
                            str_value = value.decode('utf-8', errors='ignore')
                        else:
                            str_value = str(value)
                        description_value = clean_description(str_value)
                        description_value = description_value.strip()
            else:
                print("No EXIF data found")
            
            result.append({
                "image": filename,
                "ImageDescription": description_value
            })
    output_path = os.path.join(folder_path, "total_description.json")
    print(f"\n结果已保存到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    folder = input("请输入图片文件夹路径: ")
    print_exif_for_folder(folder)
