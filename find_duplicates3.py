import json

def find_matching_items(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    partners = data['partners']['partner']
    groups = {}

    for partner in partners:
        key = (partner['inputPath'], partner['storageAccount'], partner['fileNamePattern'], partner['inputFormat'])

        if key not in groups:
            groups[key] = []

        groups[key].append(partner)

    return groups

def print_groups(groups):
    for key, items in groups.items():
        input_path, storage_account, file_pattern, input_format = key
        print(f"Group:")
        print(f"Input Path: {input_path}")
        print(f"Storage Account: {storage_account}")
        print(f"File Name Pattern: {file_pattern}")
        print(f"Input Format: {input_format}")
        print("Items:")
        for item in items:
            print(item)
        print()

json_file = 'data.json'
groups = find_matching_items(json_file)
print_groups(groups)
