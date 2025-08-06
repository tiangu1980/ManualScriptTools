import json

def merge_json_files(file1_path, file2_path, output_path):
    # 读取两个 JSON 文件
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    # 建立以 image 为主键的字典
    merged_dict = {}

    # 先插入第一个文件的数据
    for item in data1:
        image_key = item["image"]
        merged_dict[image_key] = item.copy()

    # 合并第二个文件的数据
    for item in data2:
        image_key = item["image"]
        if image_key in merged_dict:
            merged_dict[image_key].update(item)  # 合并属性
        else:
            merged_dict[image_key] = item.copy()

    # 转为列表
    merged_list = list(merged_dict.values())

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(merged_list, out, ensure_ascii=False, indent=2)

    print(f"合并完成，共 {len(merged_list)} 条记录，保存为 {output_path}")

# 示例调用
if __name__ == "__main__":
    merge_json_files("merged_total_location_v1.json", "Total_b_cleaned_removed2.json", "Total_b_full.json")
