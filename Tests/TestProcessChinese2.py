import re
from collections import Counter

def clean_text(text):
    # 去掉除了句号以外的所有符号
    cleaned_text = re.sub(r'[^\u4e00-\u9fff。]', '', text)  # 保留中文和句号
    # 将连续的多个句号只保留一个
    cleaned_text = re.sub(r'。+', '。', cleaned_text)
    return cleaned_text

def char_frequency(text):
    # 将文本转换为字符数组
    char_list = list(text)
    # 计算字符频率
    freq = Counter(char_list)
    return freq
    

def read_and_count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def main():
    # 读取文本（这里假设文本是Unicode编码的中文字符串）
    #text = "这是一个测试文本。这里有一些标点符号，！？；：、，等等。还有多个句号。。。。我们要去掉它们。"
    file_path = 'C:\ThomasTemp\易经 译文.TXT'  # 替换为你的文件路径
    text = read_and_count_characters(file_path)
    
    # 清理文本
    cleaned_text = clean_text(text)
    
    # 计算字符频率
    frequency = char_frequency(cleaned_text)
    
    # 打印字符频率
    for char, freq in frequency.items():
        print(f"{char}: {freq}")

if __name__ == "__main__":
    main()