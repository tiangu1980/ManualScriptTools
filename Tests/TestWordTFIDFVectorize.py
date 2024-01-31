# 导入所需的库
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



# TF-IDF 向量化
 #向量化是一种将文本转换为向量的方法，它考虑了词在文档中的频率（TF）和词在文档集合中的重要性（IDF），它的值等于 TF 和 IDF 的乘积，即 TF-IDF = TF * IDF。TF-IDF 的值越大，表明这个词在该文档中的重要程度越高。用 TF-IDF 构建的词袋模型可以更好地表达文本特征，TF-IDF 常被用于文本分类任务中的文本向量化表示56。
 # 词在文档集合中的重要性（IDF）是一种反映词的稀有程度的指标，它的计算公式是：
# IDF=log(N​/(n+1))
# 其中，N 是文档集合中的文档总数，n 是包含该词的文档数，分母加 1 是为了避免出现 n=0 的情况，导致 IDF 无穷大12。IDF 的值越大，说明该词在文档集合中越稀有，越能够区分不同的文档。

# 定义一个文本集合，包含 4 个文档
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

# 创建一个计数向量器，用来生成词频矩阵
vectorizer = CountVectorizer()

# 将文本集合转化为词频矩阵，每一行对应一个文档，每一列对应一个词
X = vectorizer.fit_transform(corpus)

# 创建一个 TF-IDF 转换器，用来生成 TF-IDF 值
transformer = TfidfTransformer()

# 将词频矩阵 X 统计成 TF-IDF 值
tfidf = transformer.fit_transform(X)

# 打印结果
print(tfidf.toarray())
