import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import os
import sys
import joblib
import re

model_file_extension=".joblib"

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

def train_model(train_file, model_file_arg, incols, merge, target):
    model_file=model_file_arg + model_file_extension
    # 检查模型文件是否已存在
    if os.path.exists(model_file):
        print(f"Load model {model_file}")
    
        # 如果存在，加载模型
        vectorizer, classifier = joblib.load(model_file)
        
        # 读取训练数据
        df = pd.read_excel(train_file, names=incols.append(target))

        # 填充空单元格
        df = df.fillna(' ')
        
        colName1=incols[0]
        if len(incols)>1 :
            colName2=incols[1]
            df['InText'] = df.apply(lambda row: str(row[colName1]) + merge + str(row[colName2]), axis=1)
        else:
            df['InText'] = df[colName1]
        
        df['InLable'] = df[target]

        # 对文本进行预处理
        df['InText'] = df['InText'].apply(preprocess_text)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(df['InText'], df['InLable'], test_size=0.2, random_state=42)

        # TF-IDF特征提取
        X_train_tfidf = vectorizer.transform(X_train)
        
        # 继续训练模型
        classifier.partial_fit(X_train_tfidf, y_train) #, classes=np.unique(df['InLable']))
    else:
        print(f"Create model {model_file}")
        
        # 如果模型不存在，新建模型
        df = pd.read_excel(train_file, names=incols.append(target))

        # 填充空单元格
        df = df.fillna(' ')
        
        colName1=incols[0]
        if len(incols)>1 :
            colName2=incols[1]
            df['InText'] = df.apply(lambda row: str(row[colName1]) + merge + str(row[colName2]), axis=1)
        else:
            df['InText'] = df[colName1]
        
        df['InLable'] = df[target]

        # 对文本进行预处理
        df['InText'] = df['InText'].apply(preprocess_text)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(df['InText'], df['InLable'], test_size=0.2, random_state=42)

        # TF-IDF特征提取
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # 初始化并训练多项式贝叶斯分类器
        classifier = MultinomialNB()
        classifier.fit(X_train_tfidf, y_train)

    # 保存模型
    joblib.dump((vectorizer, classifier), model_file)



def work_model(work_file, model_file_arg, incols, merge, target):
    model_file=model_file_arg + model_file_extension
    if os.path.exists(model_file):
        # 加载模型
        vectorizer, classifier = joblib.load(model_file)

        # 读取工作数据
        df_work = pd.read_excel(work_file, names=incols)

        # 填充空单元格
        df_work = df_work.fillna(' ')
        print(len(df_work))
        
        colName1=incols[0]
        if len(incols)>1 :
            colName2=incols[1]
            df_work['InText'] = df_work.apply(lambda row: str(row[colName1]) + merge + str(row[colName2]), axis=1)
        else:
            df_work['InText'] = df_work[colName1]

        # 对文本进行预处理
        df_work['InText'] = df_work['InText'].apply(preprocess_text)

        # TF-IDF特征提取
        X_work_tfidf = vectorizer.transform(df_work['InText'])
        
        # 预测标签
        predictions = classifier.predict(X_work_tfidf)

        # 打印结果
        #for i, row in df_work.iterrows():
        #    print(f"Text: {row['InText']}, Predicted Label: {predictions[i]}")
        #for i, (index, row) in enumerate(df_work.iterrows()):
        for i in range(len(predictions)):
            #print(f"Index: {index}, Text: {row['InText']}, Predicted Label: {predictions[i]}")            
            #print(f"{index}\t{predictions[i]}")
            print(f"{i}\t{predictions[i]}")
    else:
        print("Error: Model file not found. Please run in train mode first.")

# python TfIdf-PolynomialBayesianClassifier.py --mode train --infile SAT-DSAT-Lable-train.xlsx  --model SAT-DSAT-L1 --incols "SAT" "DSAT" --merge but --target L1
# python TfIdf-PolynomialBayesianClassifier.py --mode work  --infile SAT-DSAT-Lable-input.xlsx  --model SAT-DSAT-L1 --incols "SAT" "DSAT" --merge but --target L1
# python TfIdf-PolynomialBayesianClassifier.py --mode train --infile SAT-DSAT-Lable-train.xlsx  --model DSAT-L3 --incols "DSAT" --merge but --target L3
# python TfIdf-PolynomialBayesianClassifier.py --mode work  --infile SAT-DSAT-Lable-input.xlsx  --model DSAT-L3 --incols "DSAT" --merge but --target L3
if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='python TfIdf-PolynomialBayesianClassifier.py --mode train --infile DSAT-L3-train.xlsx  --model DSAT-L3 --incols "col1,col2" --merge but --target col3 \npython TfIdf-PolynomialBayesianClassifier.py --mode work  --infile DSAT-L3-input.xlsx  --model DSAT-L3 --incols "col1,col2" --merge and --target col3 ')
    parser.add_argument('--mode', type=str, help='train | work')
    parser.add_argument('--infile', type=str, help='Excel file name')
    parser.add_argument('--model', type=str, help='Model start name without extension')
    parser.add_argument('--incols', nargs='+', type=str, help='List of input columns(2)')
    parser.add_argument('--merge', type=str, help='How to merge the input columns')
    parser.add_argument('--target', type=str, help='Target column to train')
    args = parser.parse_args()
    
    infile = args.infile
    if not os.path.exists(infile):
        print("Invalid input file. It does not exist.")
        exit()
        
    incols=args.incols
    if not incols:
        print("Invalid input cols. At least 1 column name.")
        exit()
        
    merge=args.merge
    if not merge:
        merge = " and "
    if merge == "and":
        merge = " and "
    elif merge == "but":
        merge = " but other side "
    else:
        print("Invalid merge. Please enter 'and' or 'but'.")
        exit()

    target=args.target
    if not target:
        print("Invalid target. At least 1 column name.")
        exit()
    
    model_file_arg=args.model
    if not model_file_arg:
        print("Invalid model. There must be a legal model file name (without extension).")
        exit()

    mode = args.mode
    print(f" --mode {mode} --infile {infile} --model {model_file_arg} --incols{incols} --merge{merge} --target {target}")
    if mode == "train":
        train_model(infile, model_file_arg, incols, merge, target)
        print(f"Model trained and saved to {model_file_arg}")
    elif mode == "work":
        work_model(infile, model_file_arg, incols, merge, target)
    else:
        print("Invalid mode. Please enter 'train' or 'work'.")
        exit()
