
import pandas as pd
import os

paths = [
"D:\\GrowthAI\\4 ReGroupInDimensions\\Total\\total_(color_n=100_t=1e-07)_20250725_161231_785.xlsx",
"D:\\GrowthAI\\4 ReGroupInDimensions\\Total\\total_(color_n=65_t=1e-06)_20250725_161957_845.xlsx"]

def generate_summary(excel_path):
    df = pd.read_excel(excel_path)
    categories = sorted(df['Category'].value_counts().to_dict().items())
    sorted_categories = sorted(categories, key=lambda x: x[1], reverse=True)  # 按File排序

    #print("Category    File")
    for cat, count in sorted_categories:
        print(f"    {cat}    {count}")

if __name__ == "__main__":    
    for path in paths:
        if os.path.exists(path):
            print("----------------------------------")
            print(path)
            generate_summary(path)
        else:
            print(f"文件不存在: {path}")
