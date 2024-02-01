import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import sys
import joblib
import re

def preprocess_text(text):
    # 将所有换行符转换为空格
    text = str(text).replace('\n', ' ')
    
    # 将文本转换为小写
    text = text.lower()

    # 将“'s”转换为“is”
    text = re.sub(r"'s", " is", text)

    # 将“'d”转换为“would”
    text = re.sub(r"'d", " would", text)

    # 将“'m”转换为“am”
    text = re.sub(r"'m", " am", text)

    # 将“'re”转换为“are”
    text = re.sub(r"'re", " are", text)

    # 将所有非字符的符号转换为空格
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # 将字符串转换为空格： " is ", " am ", " are "
    #text = re.sub(r'\sis\s|\sam\s|\sare\s', ' ', text)

    # 将“  ”转换为“ ”
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"  ", " ", text)
    text = re.sub(r"  ", " ", text)
    
    # 将“would”转换为“will”
    text = re.sub(r'would', 'will', text)

    return text

#def train_model(train_file, model_file):
#    # 读取训练数据
#    df = pd.read_excel(train_file, names=['Text', 'L3'])
#
#    # 填充空单元格
#    df = df.fillna('No')
#
#    # 对文本进行预处理
#    df['Text'] = df['Text'].apply(preprocess_text)
#
#    # 划分训练集和测试集
#    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['L3'], test_size=0.2, random_state=42)
#
#    # TF-IDF特征提取
#    vectorizer = TfidfVectorizer()
#    X_train_tfidf = vectorizer.fit_transform(X_train)
#    
#    # 初始化并训练多项式贝叶斯分类器
#    classifier = MultinomialNB()
#    classifier.fit(X_train_tfidf, y_train)
#
#    # 保存模型
#    joblib.dump((vectorizer, classifier), model_file)

def train_model(train_file, model_file):
    # 检查模型文件是否已存在
    if os.path.exists(model_file):
        # 如果存在，加载模型
        vectorizer, classifier = joblib.load(model_file)
        
        # 读取训练数据
        df = pd.read_excel(train_file, names=['Text', 'L3'])

        # 填充空单元格
        df = df.fillna('')

        # 对文本进行预处理
        df['Text'] = df['Text'].apply(preprocess_text)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['L3'], test_size=0.2, random_state=42)

        # TF-IDF特征提取
        X_train_tfidf = vectorizer.transform(X_train)
        
        # 继续训练模型
        classifier.partial_fit(X_train_tfidf, y_train) #, classes=np.unique(df['L3']))
    else:
        # 如果模型不存在，新建模型
        df = pd.read_excel(train_file, names=['Text', 'L3'])

        # 填充空单元格
        df = df.fillna('')

        # 对文本进行预处理
        df['Text'] = df['Text'].apply(preprocess_text)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['L3'], test_size=0.2, random_state=42)

        # TF-IDF特征提取
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # 初始化并训练多项式贝叶斯分类器
        classifier = MultinomialNB()
        classifier.fit(X_train_tfidf, y_train)

    # 保存模型
    joblib.dump((vectorizer, classifier), model_file)



def work_model(work_file, model_file):
    if os.path.exists(model_file):
        # 加载模型
        vectorizer, classifier = joblib.load(model_file)

        # 读取工作数据
        df_work = pd.read_excel(work_file, names=['Text'])

        # 填充空单元格
        df_work = df_work.fillna('No')

        # 对文本进行预处理
        df_work['Text'] = df_work['Text'].apply(preprocess_text)

        # TF-IDF特征提取
        X_work_tfidf = vectorizer.transform(df_work['Text'])
        
        # 预测标签
        predictions = classifier.predict(X_work_tfidf)

        # 打印结果
        #for i, row in df_work.iterrows():
        #    print(f"Text: {row['Text']}, Predicted Label: {predictions[i]}")
        for i, (index, row) in enumerate(df_work.iterrows()):
            #print(f"Index: {index}, Text: {row['Text']}, Predicted Label: {predictions[i]}")            
            print(f"{index}\t{predictions[i]}")
    else:
        print("Error: Model file not found. Please run in train mode first.")

if __name__ == "__main__":
    #mode = input("Enter mode (train or work): ").lower()
    #input_file = input("Enter input file path (xlsx): ")
    
    if len(sys.argv)<3 :
        print("Invalid argv lengh.")
        exit()
        
    mode=sys.argv[1]
    input_file=sys.argv[2]
    
    if not os.path.exists(input_file):
        print("Data file is not exists.")
        exit()

    if mode == "train":
        model_file = "text_classifier_model.joblib"
        train_model(input_file, model_file)
        print(f"Model trained and saved to {model_file}")
    elif mode == "work":
        model_file = "text_classifier_model.joblib"
        work_model(input_file, model_file)
    else:
        print("Invalid mode. Please enter 'train' or 'work'.")
        exit()
