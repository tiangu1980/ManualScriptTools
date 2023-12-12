import json

def find_duplicates(json_data):
    duplicates = {}
    
    # 遍历partner列表
    for partner in json_data['partners']['partner']:
        input_path = partner['inputPath']
        storage_account = partner['storageAccount']
        file_name_pattern = partner['fileNamePattern']
        input_format = partner['inputFormat']
        key = (input_path, storage_account, file_name_pattern, input_format)
        
        # 检查是否已经存在相同的键
        if key in duplicates:
            duplicates[key].append(partner)
        else:
            duplicates[key] = [partner]
    
    # 打印符合条件的分组
    for key, items in duplicates.items():
        if len(items) > 1:
            print(f"Items with inputPath '{key[0]}', storageAccount '{key[1]}', fileNamePattern '{key[2]}' and inputFormat '{key[3]}' are:")
            for item in items:
                print(item)
            print()

# 读取JSON文件
with open('data.json') as f:
    json_data = json.load(f)

# 调用函数查找重复项并打印结果
find_duplicates(json_data)
