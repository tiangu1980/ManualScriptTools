import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus 
from datetime import datetime
import time
import re

# 设置 MySQL 连接参数
db_host = 'localhost'
db_user = 'root'
db_password = '0@th5tyx'
db_name = 'workspeech'

# 对密码进行URL编码  
encoded_password = quote_plus(db_password)  

# 创建MySQL连接字符串  
connection_string = f'mysql://{db_user}:{encoded_password}@{db_host}/{db_name}'  

# 创建 MySQL 连接引擎
engine = create_engine(connection_string)

# 读取 Excel 文件数据
#excel_file = 'Data insights-OpenAI data Dec-23-0131.xlsx'
#sheet_name = 'Speech 1k vs. Open AI paid'
#columns = ['Cloud Customer GUID', 'Name', 'Current Status', 'Current CU', 'CU MoM', 'Account Manager', 'Country Name', 'Industry', 'Vertical', 'Segments', 'Current Month Meters', 'Kind', 'Month']  # 假设前6列的列名为 Col1 到 Col6
#df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=columns)
#df['Month'] = df['Month'].dt.strftime("%Y_%m_%d")

# 使用正则表达式匹配字母字符
#cleaned_sheet_name = re.sub(r'[^a-zA-Z]', '', sheet_name)

# 获取当前时间戳
#timestamp = int(time.time())

# 设置输出表的名称
#output_table_name = f'{cleaned_sheet_name}_{timestamp}'

# 将数据导入到 MySQL 数据库
#df.to_sql(name=output_table_name, con=engine, if_exists='append', index=False)


df_months=pd.read_sql_query('select distinct Month from workspeech.speechkvsopenaipaid_1710827039 ', engine)
df_months = df_months.sort_values('Month')

dfs_perMonth = []
for month in df_months['Month']:
    print(month)
    sqlquery = f'SELECT Name, SUM(CASE WHEN Kind = \'AOAI\' THEN `Current CU` ELSE 0 END) AS AOAI_CU_{month}, SUM(CASE WHEN Kind = \'Speech\' THEN `Current CU` ELSE 0 END) AS Speech_CU_{month} FROM workspeech.speechkvsopenaipaid_1710744967 WHERE Month = \'{month}\' GROUP BY Name, Month'

    df_current = pd.read_sql_query(sqlquery, engine)
    dfs_perMonth.append(df_current)

merged_df = pd.concat([df.set_index('Name') for df in dfs_perMonth], axis=1, join='outer')
merged_df = merged_df.reset_index()
print(merged_df.columns)
merged_df.fillna(0, inplace=True)
merged_df.to_excel('output_Name-MandM.xlsx', index=False)
print("Data imported successfully into MySQL database.")
