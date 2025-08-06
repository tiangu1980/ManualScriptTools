import json
import argparse
import time
from llama_cpp import Llama

# 加载本地模型（路径可调整）
# HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --resume-download --local-dir-use-symlinks False TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF --include "Nous-Hermes-2-Mixtral-8x7B-DPO.Q8_0.gguf" --local-dir ./model
MODEL_PATH = "./model/Nous-Hermes-2-Mistral-7B-DPO.Q8_0.gguf"

# 初始化模型
print("🔄 Loading model...")
llm = Llama(model_path=MODEL_PATH, n_threads=8, n_ctx=2048)
print("✅ Model loaded.")

def build_prompt(theme: str) -> str:
    return f"""
You are an expert content writer for a professional stock image platform. You MUST follow the strict rules below:

1. The image theme is: "{theme}"
2. You MUST generate:
   - A SHORT English TITLE: 3 to 5 words MAXIMUM, summarizing the theme precisely and attractively.
   - A DESCRIPTION: One fluent English sentence, 15 words or fewer, NO long phrases, NO conjunctions, NO complex grammar.

STRICT FORMAT (do NOT change it):
Title: <your concise title>
Description: <your clean description>
"""

def generate_output(prompt: str, max_tokens=150, temperature=0.7):
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n"]
        )
        return output['choices'][0]['text'].strip()
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return ""

def parse_output(output: str):
    title, desc = "", ""
    try:
        lines = output.splitlines()
        for line in lines:
            if line.lower().startswith("title:"):
                title = line.partition(":")[2].strip()
            elif line.lower().startswith("description:"):
                desc = line.partition(":")[2].strip()
    except Exception as e:
        print(f"⚠️ Failed to parse output: {e}")
    return title, desc

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_data = []
    for idx, entry in enumerate(data):
        theme = entry.get("theme", "")
        if not theme:
            entry["title"] = ""
            entry["description"] = ""
            updated_data.append(entry)
            continue

        print(f"✨ [{idx+1}/{len(data)}] Processing theme: {theme}")
        prompt = build_prompt(theme)
        output = generate_output(prompt)

        if not output:
            print(f"⚠️ Skipping due to empty output for theme: {theme}")
            entry["title"] = ""
            entry["description"] = ""
        else:
            title, description = parse_output(output)
            entry["title"] = title
            entry["description"] = description

        updated_data.append(entry)
        time.sleep(0.2)  # 避免CPU过载，可调节

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Finished. Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate title and description for image themes using local LLM.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    process_file(args.input, args.output)
