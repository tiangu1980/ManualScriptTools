import os

def find_TTL_in_file(file_path):
    with open(file_path, 'r') as f:
        schema_lines = f.readlines()

    target_line = None
    for line in reversed(schema_lines[-5:]):
        if line.strip().lower().startswith("ttl "):
            target_line = line
            break
    TTLContent = ""
    # 如果找到匹配的行，则打印 "TTL " 之后的部分
    if target_line:
        ttl_index = target_line.lower().index("ttl ")
        ttl_content = target_line[ttl_index + 4:].strip()
        TTLContent = ttl_content
    else:
        TTLContent = "NoTTL"

    return TTLContent

def search_tables_in_directory(directory, table_names):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('titan_byod_prod_replica.') and file.endswith('.txt'):
                table_name = file[len('titan_byod_prod_replica.'):-len('.txt')]
                if table_name in table_names:
                    schema_file = os.path.join(root, file)
                    TTLContent = find_TTL_in_file(schema_file)
                    if TTLContent:
                        print(f'{table_name}    {TTLContent}')
                    else:
                        print(f'{table_name}    NoTTL')

# 指定目录和文件
base_dir = r'D:\DM.Titan.Backend\TableSchema\BYOD'
tables_file = 'tables.txt'

# 读取tables.txt文件中的每一行表名
with open(tables_file, 'r') as f:
    table_names = f.read().splitlines()

# 在指定目录及其子文件夹中搜索表schema文件并打印分区信息
search_tables_in_directory(base_dir, table_names)
