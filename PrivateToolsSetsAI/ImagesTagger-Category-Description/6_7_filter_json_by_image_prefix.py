
import json
import argparse

def filter_json(input_file, output_file, prefix):
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Filter out items where image starts with prefix
    if isinstance(data, list):
        filtered_data = [item for item in data if not item.get('image', '').startswith(prefix)]
    elif isinstance(data, dict):
        filtered_data = {k: v for k, v in data.items() if not v.get('image', '').startswith(prefix)}
    else:
        raise ValueError("Unsupported JSON structure")
    
    # Save the filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter JSON by image prefix')
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('output', help='Output JSON file')
    parser.add_argument('prefix', help='Prefix to filter by')
    
    args = parser.parse_args()
    filter_json(args.input, args.output, args.prefix)
