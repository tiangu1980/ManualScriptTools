import os
import sys

def main(firmedPath, tbSrcWhere, targetPath, srcCluster, targetCluster):
    # 拼接参数
    tbPath = os.path.join(firmedPath, tbSrcWhere)
    
    partPath = tbSrcWhere.split('\\', 2)[2]
    print(f"partPath         ： {partPath}")
    last_backslash_index = partPath.rfind('\\')
    result = partPath[:last_backslash_index]
    print(f"result         ： {result}")
    targetLocation = os.path.join(targetPath, result)
    os.makedirs(targetLocation, exist_ok=True)
    
    srcFileName = os.path.basename(tbSrcWhere)

    print(f"tbPath         ： {tbPath}")
    print(f"targetLocation ： {targetLocation}")
    print(f"srcFileName    ： {srcFileName}")


    # 检查文件是否存在
    if not os.path.exists(tbPath):
        print(f"文件不存在：{tbPath}")
        return

    # 读取文件内容
    with open(tbPath, 'r') as tbFile:
        tbContent = tbFile.read()

    # 检查是否包含字符串“_replica.”
    if "_replica." in tbPath:
        Is3xReplicaTable(tbContent, targetLocation, srcFileName, srcCluster, targetCluster)
    else:
        NotReplica(tbContent, targetLocation, srcFileName, srcCluster, targetCluster)

def Is3xReplicaTable(tbContent, targetLocation, srcFileName, srcCluster, targetCluster):
    # 复制文件内容到新变量
    tbReplica1 = tbContent
    tbReplica2 = tbContent

    # 替换内容
    tbReplica1 = tbReplica1.replace("_replica.", "_replica_1.")
    tbReplica1 = tbReplica1.replace(srcCluster, targetCluster)
    tbReplica1 = tbReplica1.replace("{shard_replica}", "{shard_replica_1_3x}")

    # 生成新文件路径
    targetFilePath1 = os.path.join(targetLocation, srcFileName)
    targetFilePath1 = targetFilePath1.replace("_replica.", "_replica_1.")

    # 写入新文件
    with open(targetFilePath1, 'w') as targetFile1:
        targetFile1.write(tbReplica1)

    # 类似地处理tbReplica2

    # 替换内容
    tbReplica2 = tbReplica2.replace("_replica.", "_replica_2.")
    tbReplica2 = tbReplica2.replace(srcCluster, targetCluster)
    tbReplica2 = tbReplica2.replace("{shard_replica}", "{shard_replica_2_3x}")

    # 生成新文件路径
    targetFilePath2 = os.path.join(targetLocation, srcFileName)
    targetFilePath2 = targetFilePath2.replace("_replica.", "_replica_2.")

    # 写入新文件
    with open(targetFilePath2, 'w') as targetFile2:
        targetFile2.write(tbReplica2)

def NotReplica(tbContent, targetLocation, srcFileName, srcCluster, targetCluster):
    # 替换内容
    tbContent = tbContent.replace(srcCluster, targetCluster)

    # 生成新文件路径
    targetFilePath = os.path.join(targetLocation, srcFileName)

    # 写入新文件
    with open(targetFilePath, 'w') as targetFile:
        targetFile.write(tbContent)

# "D:\\DM.Titan.Backend\\", "TableSchema\\BYOD\\ConsumerCommercial\\consumercommercialdata\\titan_byod_prod_replica.consumercommercialdata_local.txt", "D:\\DM.Titan.Backend\\TableSchema\\BYOD3X\\", "Titan_SelfServe", "Titan_SelfServe_3X"
# python Convert2xTo3xChTb.py D:\DM.Titan.Backend\  TableSchema\BYOD\ConsumerCommercial\consumercommercialdata\titan_byod_prod.consumercommercialdata.txt  D:\DM.Titan.Backend\TableSchema\BYOD3X\  Titan_SelfServe   Titan_SelfServe_3X
if __name__ == "__main__":
    # 传入五个输入参数
    # 确保至少提供了五个参数
    print(f"参数：{len(sys.argv)}")
    print(f"firmedPath：{sys.argv[1]}")
    print(f"tbSrcWhere：{sys.argv[2]}")
    print(f"targetPath：{sys.argv[3]}")
    print(f"srcCluster：{sys.argv[4]}")
    print(f"targetCluster：{sys.argv[5]}")
    if len(sys.argv) >= 6:
        firmedPath = sys.argv[1]
        tbSrcWhere = sys.argv[2]
        targetPath = sys.argv[3]
        srcCluster = sys.argv[4]
        targetCluster = sys.argv[5]
        main(firmedPath, tbSrcWhere, targetPath, srcCluster, targetCluster)
    else:
        print("请提供正确的参数")
