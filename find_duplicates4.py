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

    filtered_groups = {k: v for k, v in groups.items() if len(v) > 1}
    return filtered_groups

def print_groups(groups):
    total_groups = len(groups)
    group_number = 1

    for key, items in groups.items():
        input_path, storage_account, file_pattern, input_format = key
        dri_contacts = set()
        names = set()
        tables = set()

        print(f"Group {group_number}/{total_groups}")
        print("Filter:")
        print(f"    inputPath       = {input_path}")
        print(f"    storageAccount  = {storage_account}")
        print(f"    fileNamePattern = {file_pattern}")
        print(f"    inputFormat     = {input_format}")
        print("driContact:")

        for item in items:
            if item['driContact']:
                dri_contacts.add(item['driContact'])

        dri_contacts_combined = ';'.join(list(dri_contacts))
        print(f"    {dri_contacts_combined}")

        print("name:")
        for item in items:
            if item['name']:
                names.add(item['name'])

        for name in names:
            print(f"    {name}")

        print("table:")
        for item in items:
            if item['table']:
                tables.add(item['table'])

        for table in tables:
            print(f"    {table}")

        group_number += 1
        print()

json_file = 'data.json'
groups = find_matching_items(json_file)
print_groups(groups)
