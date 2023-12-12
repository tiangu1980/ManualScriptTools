import json

# 读取文件
data = []
with open('machines.txt', 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        columns = line.split('\t')
        item = {
            "id": str(idx + 1),
            "machine": columns[0],
            "ip": columns[2]
        }
        data.append(item)

# 输出为 JSON 格式
output = json.dumps(data, indent=2)
print(output)
