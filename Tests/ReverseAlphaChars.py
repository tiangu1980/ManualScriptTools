import os

# 这段脚本的作用是对输入的字符串进行处理，将其中的字母进行位置交换，使得最左侧的字母与最右侧的字母交换，次左侧的字母与次右侧的字母交换，依此类推，直到整个字符串被处理完毕。
inStr="a,b$c  45326 retwr,.67xf+_9q"
outStrDict={}

def isAz(ch):
    print(f"     -- isAz {ch}")
    if (((ch>='A') and (ch<='Z')) or ((ch>='a') and (ch<='z'))):
        return True
    else:
        return False

length=len(inStr)

rPtr=len(inStr)-1

for lPtr in range(length):
    print(lPtr)
    if (lPtr>=rPtr):
        print("lPtr>=rPtr, exit")
        break
    print(f"+++++ lPtr {lPtr}, rPtr {rPtr}") 
    if inStr[lPtr].isalpha():#(isAz(inStr[lPtr])):
        print(f"     -- inStr[lPtr] {inStr[lPtr]}")
        rRange=range(rPtr, lPtr, -1)# range(rPtr, lPtr)
        print(f"     -- rRange {rRange}")
        for j in rRange:
            print(f"         .. j {j} , inStr[j] {inStr[j]}")
            if (lPtr>=rPtr):
                break;
            if inStr[j].isalpha():#(isAz(inStr[j])):
                outStrDict[lPtr]=inStr[j]
                outStrDict[j]=inStr[lPtr]
                rPtr=j-1
                break
            else: 
                outStrDict[lPtr]=inStr[lPtr]
                outStrDict[j]=inStr[j]
    else:
        outStrDict[lPtr]=inStr[lPtr]

outStrDictSorted=sorted(outStrDict.items())
result_str = ''.join(f"{value}" for key, value in outStrDictSorted)
print(result_str)