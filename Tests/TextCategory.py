from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# 输入数据
texts = [
    "First paragraph of text 1.",
    "Second paragraph of text 2.",
    "Third paragraph of text 3."
]

# 对应的类别标签
labels = ["Category1", "Category2", "Category3"]

# 创建模型
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 将类别标签编码为数字
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 训练模型
model.fit(texts, encoded_labels)

# 输入新的文本进行预测
new_texts = [
    "New text for classification 1.",
    "New text for classification 2.",
    "New text for classification 3."
]

predicted_labels = model.predict(new_texts)

# 将预测的标签解码为原始类别
decoded_labels = label_encoder.inverse_transform(predicted_labels)

# 输出结果
for text, label in zip(new_texts, decoded_labels):
    print(f"Text: {text}\nPredicted Category: {label}\n{'='*30}")
