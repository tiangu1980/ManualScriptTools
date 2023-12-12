import csv
import json

def process_json_file(json_path, row_data):
    # 读取指定的 JSON 文件
    with open(json_path, 'r') as file:
        data = json.load(file)
        # 将 JSON 数据中与列名相同的属性内容填充到对应位置
        for key, value in row_data.items():
            if key in data:
                row_data[key] = data[key]
            else:
                row_data[key] = "No Content"
        return row_data

# 读取 CheckTables.csv 文件
with open('CheckTables.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    # 获取 CSV 文件的列名
    fieldnames = reader.fieldnames
    for row in reader:
        # 获取每一行的 JsonPath 内容
        json_path = row[fieldnames[0]]  # 使用正确的列名来访问数据
        # 复制当前行的数据，用于填充 JSON 内容
        row_data = row.copy()
        # 调用处理 JSON 文件的函数，获取填充后的数据
        row_data = process_json_file(json_path, row_data)

        # 输出处理后的该行内容，列与列之间添加四个空格间隔
        print("    ".join(row_data.values()))

# 示例结束
print("All files processed.")
