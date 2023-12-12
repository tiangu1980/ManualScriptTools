import json

# 假设您的JSON数据保存在一个名为 'data.json' 的文件中
with open('config.json', 'r') as f:
    data = json.load(f)

# 现在，data 是一个Python字典，您可以根据需要访问其中的内容
partners_list = data['partners']['partner']

# 遍历所有的 partner
for partner in partners_list:
    name = partner['name']
    table = partner['table']
    input_path = partner['inputPath']
    storage_account = partner['storageAccount']
    in_use = partner['inUse']
    driContact = partner['driContact']
    

    # 在这里进行您需要的处理
    # 例如，打印每个 partner 的信息
    #print(f"Partner Name: {name}")
    #print(f"Table: {table}")
    #print(f"Input Path: {input_path}")
    #print(f"Storage Account: {storage_account}")
    #print(f"In Use: {in_use}")
    print(f"{table}    {name}    {driContact}    {in_use}")
