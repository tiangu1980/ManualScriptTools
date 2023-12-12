import os
import re

# 定义函数来处理文本文件
def process_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        partition_by_found = False
        ttl_by_found = False
        part_by = None

        # 检查文件的最后6行
        for line in reversed(lines[-6:]):
            if re.match(r'^\s*$', line):  # 跳过空行
                continue

            if re.search(r'(?i)partition by ', line):
                partition_by_found = True
                part_by = re.sub(r'(?i)partition by ', '', line).strip()
                break

        # 如果未找到 "partition by"，则结束
        if not partition_by_found:
            print(f"File: {file_path}, Partition by not found, skipping...")
            return

        # 在最后6行中检查是否有 "ttl " 行
        for line in reversed(lines[-6:]):
            if re.match(r'^\s*$', line):  # 跳过空行
                continue

            if re.search(r'(?i)ttl ', line):
                ttl_by_found = True
                break

        # 如果 "ttl " 未找到，添加 "TTL {part_by} + INTERVAL 30 DAY" 行
        if not ttl_by_found:
            print(f"File: {file_path}, Adding TTL line...")
            # 找到以 "settings " 开头的行并插入新行
            for i, line in enumerate(lines):
                if re.match(r'^\s*settings\s+', line, re.IGNORECASE):
                    lines.insert(i, f"TTL {part_by} + INTERVAL 30 DAY\n")
                    break
            #lines.insert(0, f"TTL {part_by} + INTERVAL 30 DAY\n")

        # 保存修改后的文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
            print(f"File: {file_path}, Modification completed.")

    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(str(e))

# 主函数
def main():
    folder_path = r'D:\GitRepos\DM.Titan.Backend\TableSchema\BYOD'
    txt_file = "toTTL.txt"

    with open(txt_file, 'r', encoding='utf-8') as txt_file:
        str_names = txt_file.read().splitlines()

    for str_name in str_names:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 检查文件名是否包含 str_name，且扩展名是 txt，不分大小写
                if re.search(re.escape(str_name), file, re.IGNORECASE) and file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    process_text_file(file_path)

if __name__ == "__main__":
    main()
