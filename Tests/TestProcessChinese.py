import unicodedata
import collections

def is_chinese_char(char):
    """判断一个字符是否是中文字符"""
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  # CJK Unified Ideographs
        (cp >= 0x3400 and cp <= 0x4DBF) or  # CJK Unified Ideographs Extension A
        (cp >= 0x20000 and cp <= 0x2A6DF) or  # CJK Unified Ideographs Extension B
        (cp >= 0x2A700 and cp <= 0x2B73F) or  # CJK Unified Ideographs Extension C
        (cp >= 0x2B740 and cp <= 0x2B81F) or  # CJK Unified Ideographs Extension D
        (cp >= 0x2B820 and cp <= 0x2CEAF) or  # CJK Unified Ideographs Extension E
        (cp >= 0xF900 and cp <= 0xFAFF) or  # CJK Compatibility Ideographs
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  # CJK Compatibility Ideographs Supplement
        return True
    return False

def read_and_count_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 过滤掉非中文字符
    chinese_chars = [char for char in text if is_chinese_char(char)]
    
    # 计算字符频率
    char_counter = collections.Counter(chinese_chars)
    
    # 打印字符频率
    for char, count in char_counter.items():
        print(f'{char}: {count}')

# 示例使用
file_path = 'C:\ThomasTemp\易经 译文.TXT'  # 替换为你的文件路径
read_and_count_characters(file_path)