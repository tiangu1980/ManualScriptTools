import os

def find_partition_by_in_file(file_path):
    with open(file_path, 'r') as f:
        schema_lines = f.readlines()

    partition_by = None
    for line in schema_lines:
        if 'partition by' in line.lower():  # 修改：忽略大小写进行搜索
            partition_by = line.split('partition by', 1)[1].strip()
            break

    return partition_by

def search_tables_in_directory(directory, table_names):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('titan_byod_prod_replica.') and file.endswith('.txt'):
                table_name = file[len('titan_byod_prod_replica.'):-len('.txt')]
                if table_name in table_names:
                    schema_file = os.path.join(root, file)
                    partition_by = find_partition_by_in_file(schema_file)
                    if partition_by:
                        print(f'{table_name}    {partition_by}')
                    else:
                        print(f'{table_name}    Partition by not found')

# 指定目录和文件
base_dir = r'D:\GitRepos\DM.Titan.Backend\TableSchema\BYOD'
tables_file = 'tables.txt'

# 读取tables.txt文件中的每一行表名
with open(tables_file, 'r') as f:
    table_names = f.read().splitlines()

# 在指定目录及其子文件夹中搜索表schema文件并打印分区信息
search_tables_in_directory(base_dir, table_names)
