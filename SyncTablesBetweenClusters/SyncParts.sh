#!/bin/bash

cur_tb=$1
cur_partby=$2

# SyncParts.sh
# 读取target.txt的第一行内容并保存到target_Node变量
target_Node=$(head -n 1 targetNode.txt)
target_Path="/mnt/titan/${cur_tb}"

sudo sshpass -p "CsAdmin@2020" ssh TitanAdmin@${target_Node} "mkdir -p ${target_Path}"

# 读取part.txt的每一行
while IFS= read -r cur_Part; do
  # 创建临时文件名 cur_par_file
  cur_par_file="${cur_Part}-${cur_tb}.parquet"

  # 执行ClickHouse查询
  query="SELECT * FROM bing_prod.${cur_tb} WHERE ${cur_partby}='${cur_Part}' INTO OUTFILE '${cur_par_file}' FORMAT Parquet"
  sudo clickhouse-client --query="$query"

  echo "$(date -u)    Exported ${cur_par_file}"

  # 执行rsync，如果失败则重试
  while true; do
    sudo sshpass -p "CsAdmin@2020" rsync --partial --progress -e ssh ./"${cur_par_file}" "TitanAdmin@${target_Node}:${target_Path}"
    if [ $? -eq 0 ]; then
      break
    fi
  done

  # 清理临时文件
  echo "$(date -u)    Synced ${cur_par_file}"
  rm -f "${cur_par_file}"

done < parts.txt