
import os
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TitleGenerator:
    def __init__(self, model_path="phi-3-mini"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            truncation_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
    def generate_short_title(self, description):
        try:
            prompt = f"""Please summarize the following content using 3-7 keywords, requirements:
1. Extract core entities and actions
2. Remove modifiers and retain the backbone
3. Separate with Chinese commas
            
To be summarized: {description}
            
key word:"""
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(":")[-1].strip()[:50]
        except Exception as e:
            print(f"生成标题时出错: {str(e)}")
            return description[:50]

def process_csv(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    generator = TitleGenerator()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)

        modified = False
        for row in rows:
            if len(row) < 3: continue
            desc, headline, title = row[:3]
            
            if not headline.strip():
                row[1] = generator.generate_short_title(desc)
                modified = True
            if not title.strip():
                row[2] = generator.generate_short_title(desc)
                modified = True
            
            if modified:
                print(f"已处理: {desc[:30]}... → {row[1]} | {row[2]}")

        if modified:
            backup_path = file_path + ".bak"
            os.rename(file_path, backup_path)
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            print(f"修改已保存，原文件备份为 {backup_path}")
    except Exception as e:
        print(f"处理CSV时发生错误: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python csv_title_generator.py 输入文件.csv")
        sys.exit(1)
    process_csv(sys.argv[1])
