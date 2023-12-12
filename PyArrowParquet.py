import pyarrow.parquet as pq

parquet_file = pq.ParquetFile('D:\Dev\CheckParquet\NewOEMFeedback_W1_2022_07_27_gzip.parquet')
parquet_file.schema


table = pq.read_table('/mnt/d/Dev/CheckParquet/NewOEMFeedback_W1_2022_07_27_gzip.parquet', columns=[])
print(table.num_rows)


table2 = pq.read_table('/mnt/d/Dev/CheckParquet/NewOEMFeedback_W1_2022_08_03_gzip.parquet', columns=[])
print(table2.num_rows)

table3 = pq.read_table('/mnt/d/Dev/CheckParquet/NewOEMFeedback_W1_2022_08_10_gzip.parquet', columns=[])
print(table3.num_rows)
