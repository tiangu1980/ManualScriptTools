import os
import json

# 读取to_get_file_pattern.txt文件，获取所有需要匹配的"table"属性值
with open("to_get_file_pattern.txt", "r") as file:
    table_list = [line.strip() for line in file if line.strip()]

# 读取json item列表文件，并遍历其中的item
root_dir = "D:/temp/config (1).json"
with open(root_dir, "r") as json_file:
    data = json.load(json_file)
    partner_list = data.get("partners", {}).get("partner", [])
    for partner in partner_list:
        table_name = partner.get("table", "")
        if table_name in table_list:
            file_name_pattern = partner.get("fileNamePattern", "")
            print(f"{table_name}    {file_name_pattern}")
