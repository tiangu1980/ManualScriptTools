
import os
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', '--excel', dest='excel', type=str, required=True, help='Excel文件路径')
    parser.add_argument('--c', '--category', dest='category', type=int, required=True, help='要过滤的Category值')
    parser.add_argument("--t", '--target_dir', dest='target_dir', type=str, required=True, help='目标文件夹路径')
    args = parser.parse_args()

    # 读取Excel数据
    df = pd.read_excel(args.excel)
    
    # 获取当前目录
    current_dir = Path(args.excel).parent
    
    # 创建目标目录
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 遍历符合Category条件的文件
    for _, row in df[df['Category'] == args.category].iterrows():
        file_path = current_dir / row['File']
        if file_path.exists():
            file_path.rename(target_dir / file_path.name)
            print(f"已移动: {file_path.name}")

if __name__ == "__main__":
    main()
