import os
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

def print_exif_for_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            filepath = os.path.join(folder_path, filename)
            exif = get_exif_data(filepath)
            print(f"\n=== {filepath} ===")
            if exif:
                for tag, value in exif.items():
                    print(f"{tag}: {value}")
            else:
                print("No EXIF data found")

if __name__ == "__main__":
    folder = input("请输入图片文件夹路径: ")
    #input("请输入图片文件夹路径: ")
    print_exif_for_folder(folder)
