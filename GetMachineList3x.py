import json

# 读取文件
data = []
with open('machines.txt', 'r') as f:
    lines = f.readlines()
    total_lines = len(lines)
    group_size = 3
    num_groups = total_lines // group_size
    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_lines = lines[start_idx:end_idx]
        group_items = []
        for j, line in enumerate(group_lines):
            line = line.strip()
            columns = line.split('\t')
            item = {
                "id": str(start_idx + j + 1),
                "machine": columns[0],
                "ip": columns[2],
                "replica_id_1": str(start_idx + (j + 1) % group_size + 1),
                "replica_machine_1": group_lines[(j + 1) % group_size].split('\t')[0],
                "replica_id_2": str(start_idx + (j + 2) % group_size + 1),
                "replica_machine_2": group_lines[(j + 2) % group_size].split('\t')[0]
            }
            group_items.append(item)
        data.extend(group_items)

# 输出为 JSON 格式
output = json.dumps(data, indent=2)
print(output)
