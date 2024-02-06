# 导入所需的库
import numpy as np
import gensim
from gensim.models import Word2Vec

# 词嵌入向量化是一种将自然语言中的词转换为数学上的向量的方法，以便于进行计算和分析。
# 词嵌入向量化的目的是将一个维数为所有词的数量的高维空间嵌入到一个维数低得多的连续向量空间中，每个单词或词组被映射为实数域上的向量。
# 词嵌入向量化的优点是可以反映词之间的相似性和语义关系，而不是像独热编码或整数编码那样，只是对词进行简单的标识。
# 
# 词嵌入向量化的方法有很多，其中最常见的有以下几种：
# 
# Word2Vec：这是谷歌提出的一种词嵌入向量化的工具或算法集合，它采用了两种模型（CBOW 和 Skip-gram）和两种方法（负采样和层次 softmax）来训练词向量。Word2Vec 的核心思想是利用词的上下文信息来学习词的表示，即“一个词的含义由它周围的词来决定”。
# GloVe：这是斯坦福大学提出的一种词嵌入向量化的算法，它结合了 Word2Vec 的局部上下文信息和全局词频信息，通过最小化一个基于共现矩阵的目标函数来学习词向量。GloVe 的核心思想是利用词与词之间的共现关系来学习词的表示，即“一个词的含义由它与其他词的关系来决定”。
# FastText：这是 Facebook 提出的一种词嵌入向量化的工具或算法，它在 Word2Vec 的基础上，引入了子词（subword）的概念，即将一个词分解为更小的字符单元，如 n-gram 或字母。FastText 的核心思想是利用词的内部结构信息来学习词的表示，即“一个词的含义由它的构成部分来决定”。

# eg 1
# 定义一个词典，包含 10 个词
vocab = ["apple", "banana", "orange", "pear", "grape", "watermelon", "lemon", "lime", "cherry", "strawberry"]

# 定义一个词，要向量化的目标
word = "apple"

# 用词嵌入来向量化
# 创建一个词嵌入模型，用来生成词向量
model = Word2Vec(sentences=[vocab], vector_size=10, min_count=1)
print(f"model： {model}")
# 根据目标词取出对应的向量，即词嵌入
embedding = model.wv[word]
# 打印结果
print("Word embedding of word", word, "is:", embedding)
