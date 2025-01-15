import os
import re
from collections import Counter

# N-GRAM 模型是一种基于统计的语言模型，它假设一个词的出现只与前面 N 个词有关，而与其他任何词都不相关，这称为 N 阶马尔可夫性质。N-GRAM 模型的目的是根据历史信息来预测下一个词的概率分布，或者评估一个句子的合理性。N-GRAM 模型的参数是由语料库中的词频统计得到的，常用的有一元模型（Unigram），二元模型（Bigram），三元模型（Trigram）等，其中 N 的取值越大，模型越能捕捉语言的复杂性，但也越容易导致数据稀疏和过拟合的问题12。
# 下面是一个用 N-GRAM 模型来评估两个句子的合理性的例子3：
# 假设有一个语料库，包含以下四个句子：
# 
# This is the first document.
# This is the second second document.
# And the third one.
# Is this the first document?
# 
# 我们用 Bigram 模型来计算两个句子的概率：
# 
# s1 = “This is the first document.”
# s2 = “Want I english food.”
# 
# 首先，我们需要统计语料库中每个词和每个 Bigram 出现的次数，如下表所示：
# 
# Word	Count	Bigram	Count
# <s>	4	<s> This	2
# This	4	This is	3
# is	4	is the	2
# the	4	the first	2
# first	2	first document	2
# document	4	document .	2
# .	4	. </s>	4
# second	2	<s> And	1
# And	1	And the	1
# one	1	the second	1
# third	1	second second	1
# want	0	second document	1
# I	0	the third	1
# english	0	third one	1
# food	0	one .	1
# 		<s> Is	1
# 		Is this	1
# 
# 然后，我们用以下公式来计算每个 Bigram 的概率：
# P(wi​∣wi−1​)=count(w(i−1)​,w(i)​)​/count(w(i-1))
# 例如，P(document∣first)=count(first,document)/count(first)​=2/2​=1
# 最后，我们用以下公式来计算每个句子的概率：
# P(s)=P(w1​∣<s>)P(w2​∣w1​)P(w3​∣w2​)...P(</s>∣wn​)
# 例如，P(s1)=P(This∣<s>)P(is∣This)P(the∣is)P(first∣the)P(document∣first)P(</s>∣document)=(2/4)​×(3/4)​×(2/4)​×(2/4)​×(2/2)​×(2/4)​=0.003
# 同理，P(s2)=P(Want∣<s>)P(I∣Want)P(english∣I)P(food∣english)P(</s>∣food)=(0/4)​×(0/0)​×(0/0)​×(0/0)​×(0/0)​=0
# 由此可见，P(s1)>P(s2)，说明 s1 更合理，也符合我们的直觉。

def clean_text(text):
    # 去掉除了句号以外的所有符号
    cleaned_text = re.sub(r'[^\u4e00-\u9fff。]', '', text)  # 保留中文和句号
    # 将连续的多个句号只保留一个
    cleaned_text = re.sub(r'。+', '。', cleaned_text)
    return cleaned_text

# 打开本地文本文件（假设文件名为example.txt）
file_path = 'C:\ThomasTemp\镇妖博物馆.txt'

# 使用 with 语句确保文件操作完成后正确关闭文件
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取文件内容到一个字符串变量
    document = file.read()
    
document=clean_text(document)
corpus=list(document)

s=set(corpus)
print(s)

d=dict()
for i in s:
    d[i]={}

for i in range(len(corpus)-1):
    if corpus[i+1] in d[corpus[i]]:
        d[corpus[i]][corpus[i+1]]+=1
    else:
        d[corpus[i]][corpus[i+1]]=1

print(d)

def prob(d:dict, word):
    if word in d:
        sort_list=list(sorted(d[word].items(),key=lambda item:item[1], reverse=True))
        deno=0
        nomi=sort_list[0][1]
        for i in sort_list:
            deno+=i[1]
        return "{}".format(sort_list[0][0]) #"{}({})".format(sort_list[0][0], nomi/deno)
    else:
        return False

def checkDup(num:int, listToCheck):
    lenList=len(listToCheck)
    print(f"...checkDup {num} {lenList}")
    if lenList<num*2 :
        return False
    dupCount=0
    for i in range(lenList -1, lenList - num -1, -1):
        #print(f"...   {i}")
        if listToCheck[i]==listToCheck[i-num] :
            dupCount+=1
    if  dupCount==num :
        return False
    else:
        return True
        

def checkDups(num:int, listToCheck):
    lenList=len(listToCheck)
    if lenList<num :
        return True
    passCount=0
    for i in range(lenList -1, lenList - num -1, -1):
        if checkDup(i, outList) :
            passCount+=1
    if passCount==num :
        return True
    else:
        return False

word=input("Your word:")
outList=[]
outList.append(word)

while(True):
    print(f"---{word}")
    probword=prob(d, word)
    if (probword==False):
        break
    outList.append(probword)
    if checkDups(15, outList):
        word=probword
    else:
        break

result_str = ' '.join(f"{value}" for value in outList)
print(result_str)

#print(checkDup(1, outList))
#print(checkDup(2, outList))
#print(checkDup(3, outList))

