#!/bin/bash

# Local Table Name
cur_tb=$1
cur_size_limit=$2

start_time=$(date +%s)
end_time=$((start_time + 24 * 60 * 60))
cd ${cur_tb}

# IngestLocalParts.sh
while [ $(date +%s) -lt $end_time ]; do
  # Get parquets
  for file in *.parquet; do
    # Get Size
    file_size=$(stat -c %s "$file")

    # check file size bigger than 2496 bytes
    if [ "$file_size" -gt ${cur_size_limit} ]; then
      echo "$(date -u)    Ingest： $file"

      # 执行操作
      query="INSERT INTO bing_prod.${cur_tb} SETTINGS input_format_parquet_allow_missing_columns = 1 FORMAT Parquet"
      cat $file | clickhouse-client --query="$query"
      echo "$(date -u)    Finished： $file"
    else
      echo "$(date -u)    Skip： $file， ${file_size}"
    fi

    sleep 1
    rm -f "$file"
    echo "$(date -u)    Clean： $file， ${file_size}"
    sleep 1
  done
done

echo "$(date -u)    Script running finished."