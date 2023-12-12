import sys
import subprocess

# 定义要执行的脚本和参数
script_path = "Convert2xTo3xChTb.py"


def callConvertSingle(file_path, firmedPath, targetPath, srcCluster, targetCluster):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.rstrip('\n')  # 去除换行符
                #print(f"{line}\t{len(line.encode('utf-8'))} bytes")
                cmd = ["python", script_path, firmedPath, line, targetPath, srcCluster, targetCluster]
                print(f"cmd：  {cmd}")
                subprocess.run(cmd)
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")


# python Convert2xTo3xChTbBatch.py D:\DM.Titan.Backend\TableSchema\BYOD3X\tablesToConvert.txt D:\DM.Titan.Backend\  D:\DM.Titan.Backend\TableSchema\BYOD3X\  Titan_SelfServe   Titan_SelfServe_3X
if __name__ == "__main__":
    # 确保提供了正确的参数
    if len(sys.argv) >= 6:
        file_path = sys.argv[1]
        firmedPath = sys.argv[2]
        targetPath = sys.argv[3]
        srcCluster = sys.argv[4]
        targetCluster = sys.argv[5]
        callConvertSingle(file_path, firmedPath, targetPath, srcCluster, targetCluster)
    else:
        print("请提供文本文件的绝对路径作为参数")
