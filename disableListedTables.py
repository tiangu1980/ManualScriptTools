import os
import json

folder_path = "D:\GitRepos\DM.Titan.Backend\BringYourOwnData"  # 替换为文件夹路径

# 读取要查找的表名列表
with open("partner_list.txt", "r") as f:
    table_list = [line.strip() for line in f.readlines()]

# 遍历文件夹及子文件夹中的JSON文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)

            # 读取JSON文件内容
            with open(file_path, "r") as json_file:
                data = json.load(json_file)

            # 检查JSON数据是否以单个项的形式存储
            if isinstance(data, dict) and "table" in data:
                if data["table"] in table_list:
                    print(f"{data['table']}    {file}")
                    data["inUse"] = "false"
                    with open(file_path, "w") as json_file:
                        json.dump(data, json_file, indent=4)
                    print(f"    Done: {data['table']}")

            # 检查JSON数据是否以多个项的形式存储
            elif isinstance(data, dict) and "partners" in data and "partner" in data["partners"]:
                for item in data["partners"]["partner"]:
                    if isinstance(item, dict) and "table" in item:
                        if item["table"] in table_list:
                            print(f"{item['table']}    {file}")
                            item["inUse"] = "false"
                            with open(file_path, "w") as json_file:
                                json.dump(data, json_file, indent=4)
                            print(f"    Done: {item['table']}")
