import random

def rearrange_lines(input_file):
    lines = []
    with open(input_file, 'r') as file:
        for line in file:
            lines.append(line.strip().split())

    # 通过scale值对行进行排序
    sorted_lines = sorted(lines, key=lambda x: x[1])

    # 检查是否有任意三行的scale值相同，如果有，则重新随机排列
    while any(sorted_lines[i][1] == sorted_lines[i + 1][1] == sorted_lines[i + 2][1] for i in range(len(sorted_lines) - 2)):
        random.shuffle(sorted_lines)

    # 构建重新排序后的文件内容
    rearranged_content = '\n'.join(['\t'.join(line) for line in sorted_lines])

    return rearranged_content

# 提供输入文件的路径
input_file_path = 'machines.txt'

# 对行进行重新排列
rearranged_content = rearrange_lines(input_file_path)

# 输出重新排序后的文件内容
print(rearranged_content)
