import pandas as pd
import argparse


def set_is_surveyed_or_interviewed(row):
    is_surveyed = str(row['IsSurveyed(Yes)']).lower()
    is_interviewed = str(row['IsInterviewed (Yes)']).lower()
    if is_surveyed in ['yes', 'y'] or is_interviewed in ['yes', 'y']:
        return 'Y'
    else:
        return ''


def set_is_yes_or_y(row):
    is_yes_or_y = str(row['Scenario_w/o Searched']).lower()
    if is_yes_or_y not in ['yes', 'y']:
        return ''
    else:
        return 'Y'

# 解析命令行参数
parser = argparse.ArgumentParser(description='python 1_Scenarios.py --f1 month.xlsx --s1 Sheet1 --f2 scenario.xlsx --s2 Customers --f3 2024-02-10K.xlsx --s3 Export')
parser.add_argument('--f1', help='month.xlsx')
parser.add_argument('--s1', help='Sheet1')
parser.add_argument('--f2', help='scenario.xlsx')
parser.add_argument('--s2', help='Customers')
parser.add_argument('--f3', help='2023-02-10K.xlsx')
parser.add_argument('--s3', help='Export')
args = parser.parse_args()

# 读取数据到 df1 和 df2
df1 = pd.read_excel(args.f1, sheet_name=args.s1)
df2 = pd.read_excel(args.f2, sheet_name=args.s2)
df3 = pd.read_excel(args.f3, sheet_name=args.s3)


# 遍历 df3 的每一行， 使用month填充。
for index, row in df3.iterrows():
    if pd.isnull(row['Scenario_w/o Searched']) or pd.isnull(row['Sub Scenario_w/o Searched']):
        # 找到匹配行
        matching_rows = df1[df1['Cloud Customer GUID'] == row['Cloud Customer GUID']]
        if not matching_rows.empty:
            # 如果有多个匹配行，按照行号从大到小排序
            matching_rows.sort_index(ascending=False, inplace=True)
            # 找到第一个 'Scenario L1' 列的值不为空的行
            SpeechL1Scenario_values = matching_rows['Scenario L1'].dropna()
            # 找到第一个 'Scenario L2' 列的值不为空的行
            SpeechL2Scenario_values = matching_rows['Scenario L2'].dropna()
            if not SpeechL1Scenario_values.empty:
                df3.at[index, 'Scenario_w/o Searched'] = SpeechL1Scenario_values.iloc[0]
            if not SpeechL2Scenario_values.empty:
                df3.at[index, 'Sub Scenario_w/o Searched'] = SpeechL2Scenario_values.iloc[0]


# 遍历 df3 的每一行， 使用scenario填充。
for index, row in df3.iterrows():
    if pd.isnull(row['Scenario_w/o Searched']) or pd.isnull(row['Sub Scenario_w/o Searched']):
        # 找到匹配行
        matching_rows = df2[df2['Customer Name'] == row['Name']]
        if not matching_rows.empty:
            # 如果有多个匹配行，按照行号从大到小排序
            matching_rows.sort_index(ascending=False, inplace=True)
            # 找到第一个 'SpeechL1Scenario' 列的值不为空的行
            SpeechL1Scenario_values = matching_rows['SpeechL1Scenario'].dropna()
            # 找到第一个 'SpeechL2Scenario' 列的值不为空的行
            SpeechL2Scenario_values = matching_rows['SpeechL2Scenario'].dropna()
            if not SpeechL1Scenario_values.empty:
                df3.at[index, 'Scenario_w/o Searched'] = SpeechL1Scenario_values.iloc[0]
            if not SpeechL2Scenario_values.empty:
                df3.at[index, 'Sub Scenario_w/o Searched'] = SpeechL2Scenario_values.iloc[0]


# 保存 df3 为 原文件
df3.to_excel('out_' + args.f3, index=False)
