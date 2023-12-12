import pandas as pd
import datetime
import akshare as ak

# 指定要下载的股票代码
stock_code = '000001'  # 以平安银行（股票代码：000001）为例

# 获取当前日期
yesterday = datetime.date(2023, 6, 1).strftime('%Y-%m-%d')
today = datetime.date.today().strftime('%Y-%m-%d')

# 使用 akshare 库获取股票数据
stock_data = ak.stock_zh_a_hist(symbol=stock_code, start_date=yesterday, end_date=today, adjust='qfq')

# 将数据保存为xls文件
file_path = f'{stock_code}_{today}.xlsx'
stock_data.to_excel(file_path, index=False)
print(f'股票历史数据已保存为：{file_path}')
