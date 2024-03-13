import pandas as pd
import yfinance as yf

# 获取数据
data = yf.download('600233.SS', start='2023-01-01')

# 保存到Excel文件
data.to_excel('600233.xlsx')
