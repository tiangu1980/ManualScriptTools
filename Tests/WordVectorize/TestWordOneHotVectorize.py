# 导入所需的库
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

# 定义一个词典，包含 10 个词
vocab = ["apple", "banana", "orange", "pear", "grape", "watermelon", "lemon", "lime", "cherry", "strawberry"]

# 定义一个词，要向量化的目标
word = "apple"

# 用独热编码来向量化
# “独热编码”的英文名是One-hot encoding，它是一种将离散特征转换为向量的方法，每个特征对应一个长度为词典大小的向量，其中只有一个元素为 1，其他元素为 0。它和 N-gram 的区别是，N-gram 是一种基于统计的文本表示方法，它将文本分割成单词或字符的 N 个连续的组合，每个 N-gram 也可以用一个向量来表示，但是 N-gram 可以捕捉词的上下文信息，而 One-hot encoding 不能34。
# 创建一个计数向量器，用来生成词频矩阵
vectorizer = CountVectorizer(vocabulary=vocab)
print(f"vectorizer： {vectorizer}")
# 将词典转化为词频矩阵，每一行对应一个词，每一列对应一个词典中的词
matrix = vectorizer.fit_transform(vocab)
print(f"matrix： {matrix}")
# 将词频矩阵转化为数组，方便操作
array = matrix.toarray()
print(f"array： {array}")
# 找到目标词在词典中的索引
index = vocab.index(word)
print(f"index： {index}")
# 根据索引取出对应的向量，即独热编码
one_hot = array[index]
# 打印结果
print("One-hot encoding of word", word, "is:", one_hot)
