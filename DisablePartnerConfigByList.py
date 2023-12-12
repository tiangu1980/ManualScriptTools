import json
import os

# 指定 txt 文件路径
txt_file_path = 'file_list.txt'

# 指定 JSON 文件所在目录
json_files_directory = 'D:/GitRepos/DM.Titan.Backend/BringYourOwnData/config'

# 读取 txt 文件中的 JSON 文件名
with open(txt_file_path, 'r') as file:
    json_file_names = file.read().splitlines()

# 遍历每个 JSON 文件名
for file_name in json_file_names:
    # 构建 JSON 文件路径
    json_file_path = os.path.join(json_files_directory, file_name)
    
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
        
    # 修改 "inUse" 属性的值
    json_data['inUse'] = 'false'
    
    # 保存修改后的 JSON 文件
    with open(json_file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

print("修改完成！")
