import json

def generate_relationship(input_file):
    lines = []
    with open(input_file, 'r') as file:
        for line in file:
            lines.append(line.strip().split())

    relationships = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            relationship = {
                "id": str(i + 1),
                "machine": lines[i][0],
                "ip": lines[i][2],
                "replica_id_1": str(i + 2),
                "replica_machine_1": lines[i + 1][0]
            }
            relationships.append(relationship)

            relationship = {
                "id": str(i + 2),
                "machine": lines[i + 1][0],
                "ip": lines[i + 1][2],
                "replica_id_1": str(i + 1),
                "replica_machine_1": lines[i][0]
            }
            relationships.append(relationship)

    return relationships

# 提供输入文件的路径
input_file_path = 'machines.txt'

# 生成关系数据
relationships = generate_relationship(input_file_path)

# 输出为JSON文件
#output_file_path = 'path/to/output/file.json'
#with open(output_file_path, 'w') as file:
#    json.dump(relationships, file, indent=2)

# 输出为JSON
json_output = json.dumps(relationships, indent=2)
print(json_output)
