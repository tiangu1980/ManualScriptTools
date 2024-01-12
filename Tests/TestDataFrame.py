import pandas as pd

# 读取Excel文件到DataFrame
df = pd.read_excel("TestDataFrameSource1.xlsx")

# 获取DataFrame的列名
column_names = df.columns
print("列名:", column_names)

# 获取数据的第一行第二列的值
value_at_first_row_second_column = df.iloc[0, 1]
print("第一行第二列的值:", value_at_first_row_second_column)

# 删除列 "High"
df = df.drop(columns=["High"])

# 修改列 "Close" 的名字为 "600233"
df = df.rename(columns={"Close": "600233"})

# 从另一个DataFrame df2 中获取 "Close" 列并改名为 "600225"，然后添加到 df
df2 = pd.read_excel("TestDataFrameSource2.xlsx")  # 替换为实际的文件名
close_column_from_df2 = df2["Close"].rename("600225")
df = pd.concat([df, close_column_from_df2], axis=1)

# 获取数据的第一行第一列的值
value_at_first_row_second_column = df.iloc[0, 0]
print("第一行第一列的值:", value_at_first_row_second_column)

# 打印修改后的DataFrame
print("合并后的df")
print(df)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by="Date", ascending=False)
print("日期降序排列的df")
print(df)

df = df.sort_values(by="Date")
print("日期升序排列的df")
print(df)

# 使用条件筛选创建新的 DataFrame dfPart
dfPart = df[(df['Date'] >= '2024-01-02') & (df['Date'] <= '2024-01-04')]

# 打印新的 DataFrame dfPart
print("df片段的dfPart")
print(dfPart)


