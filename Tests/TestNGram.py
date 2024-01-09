import os

# 打开本地文本文件（假设文件名为example.txt）
file_path = 'From Max Weber Essays in sociology.txt'

# 使用 with 语句确保文件操作完成后正确关闭文件
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取文件内容到一个字符串变量
    document = file.read()
    
#document="Hi Piyush Behre, Issac Alphonso, Sharman Tan, Nick Kibre, Happy New Year. We have a team member replacement. From this month Liang liang has changed group to product team. Thomas is our new team member. He is experienced developer and will replace Liangliang's position. Thanks a lot."
document=document.replace(',',' ')
document=document.replace('.',' ')
document=document.replace('\'',' ')
document=document.replace('\‘',' ')
document=document.replace('’',' ')
document=document.replace('(',' ')
document=document.replace(')',' ')
document=document.replace('<',' ')
document=document.replace('>',' ')
document=document.replace('/',' ')
document=document.replace('\r',' ')
document=document.replace('\n',' ')
document=document.replace(':',' ')
document=document.replace('-','')
document=document.replace('  ',' ')
document=document.replace('  ',' ')
document=document.replace('  ',' ')
document=document.lower()
corpus=document.split(' ')
corpus.append(" ")

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
    if lenList<5 :
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
    if checkDups(5, outList):
        word=probword
    else:
        break

result_str = ' '.join(f"{value}" for value in outList)
print(result_str)

#print(checkDup(1, outList))
#print(checkDup(2, outList))
#print(checkDup(3, outList))

