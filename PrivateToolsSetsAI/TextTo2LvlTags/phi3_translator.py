
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def translate(text, target_lang):
    # 加载本地模型和tokenizer
    model_path = "./phi-3-mini"  # 替换为你的模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # 构建翻译提示
    prompt = f"Translate the following text to {target_lang}: {text}"
    
    # 生成翻译
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    
    # 解码并返回结果
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation.replace(prompt, "").strip()

if __name__ == "__main__":
    input_text = input("请输入要翻译的文本: ")
    target_language = input("请输入目标语言(如'法语'): ")
    result = translate(input_text, target_language)
    print(f"翻译结果: {result}")
