def get_is_issue_value(folder_pattern):
    if not folder_pattern.startswith("^") and not folder_pattern.endswith("$"):
        return 3
    elif not folder_pattern.startswith("^"):
        return 1
    elif not folder_pattern.endswith("$"):
        return 2
    else:
        return 0

file_path = "FolderPatterns.txt"  # 将your_file_path.txt替换为文件的实际路径

with open(file_path, 'r') as file:
    for line in file:
        # 使用strip()方法去除字符串中的换行符
        line = line.strip()
        # 查找FolderPattern的位置
        folder_pattern_index = line.find("\\\"FolderPattern\\\":")
        if folder_pattern_index != -1:
            # 截取FolderPattern的内容
            folder_pattern_start = line.find("\\\"", folder_pattern_index + len("\\\"FolderPattern\\\":"))
            folder_pattern_end = line.find("\\\"", folder_pattern_start + 1)
            folder_pattern = line[folder_pattern_start + 2:folder_pattern_end]

            # 获取isIssue的值
            is_issue = get_is_issue_value(folder_pattern)
            # 在每一行的末尾输出"isIssue"的值
            #print(line, "            isIssue:", is_issue)
            print(folder_pattern, "            isIssue:", is_issue)
        else:
            print("Error parsing JSON data:", line)
