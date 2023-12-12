import json

def remove_items_by_table_name(input_file, table_list_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    with open(table_list_file, 'r') as table_file:
        table_list = [line.strip() for line in table_file]

    if "partners" in data and "partner" in data["partners"]:
        data["partners"]["partner"] = [partner for partner in data["partners"]["partner"] if partner["table"] not in table_list]

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print("Done: 修改完成")

if __name__ == "__main__":
    json_file_path = "D:\\temp\\config.json"
    table_list_file_path = "D:\\temp\\table_list.txt"
    output_file_path = "D:\\temp\\config_modified.json"

    remove_items_by_table_name(json_file_path, table_list_file_path, output_file_path)
