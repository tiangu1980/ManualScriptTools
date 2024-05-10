import pandas as pd
import re
from tabulate import tabulate

def clear_characters(text):
    return re.sub('\W', '', text)

def process_column_content(text):
    # 使用空格将字符串分割成列表
    string_list = re.split(r'\W+', text) #text.split()

    # 对列表中的每个元素应用 clear_characters 函数
    #string_list = [clear_characters(element) for element in string_list]
    string_list = [element.strip() for element in string_list]

    # 对列表中的每个元素进行处理，使第一个字符大写（如果不是空字符串）
    string_list = [element.capitalize() if element and element[0].isalpha() else element for element in string_list]

    # 用空格将处理后的列表内容连接起来，并移除最后一个字符是空格的情况
    processed_text = ' '.join(string_list).strip()

    return processed_text

def calculate_ratio(num1, num2):
    # 检查 num1 是否为空，如果为空，则设为0
    num1 = 0 if num1 is None else num1
    
    # 检查 num2 是否为0或为空
    if num2 is None or num2 == 0:
        return "#DIV/0!"
    
    # 执行除法运算并格式化结果为小数点后4位
    result = round((num1 - num2) / num2, 4)
    return result

fileList = ["out_2023-4.xlsx", "out_2023-5.xlsx", "out_2023-6.xlsx", "out_2023-7.xlsx", "out_2023-8.xlsx", "out_2023-9.xlsx", "out_2023-10.xlsx", "out_2023-11.xlsx", "out_2023-12.xlsx", "out_2024-01.xlsx", "out_2024-02.xlsx", "out_2024-03.xlsx"]
sheetName = "Sheet1"
record_status = 'Current Status'

ppt_ordered_Scenarios = ["Caption", "Transcription", "Voice Agent", "Content Reader", "Language Learning", "Speech To Speech Translation", "Blank"]

Scenario_Lable = "Scenario_w/o Searched"
Subject_Lable = "Sub Scenario_w/o Searched"

monthsDfs = []
ls_allMainDf = []
ls_allSubDf = []
fileList.sort()
# 将每个文件读入dataframe, 进行数据整理后计算各个Scenario的CU和CCID的总值，并contact到一个新的dataframe中
for file in fileList:
    monthDf = pd.read_excel(file, sheet_name=sheetName)
    monthDf[Scenario_Lable].fillna("(blank)", inplace=True)
    monthDf[Subject_Lable].fillna("(blank)", inplace=True)
    monthDf['Name'].fillna(monthDf["Cloud Customer GUID"], inplace=True)
    monthDf['Current CU'].fillna(0, inplace=True)
    monthDf[Scenario_Lable] = monthDf[Scenario_Lable].apply(process_column_content)
    monthDf[Subject_Lable] = monthDf[Subject_Lable].apply(process_column_content)
    monthDf = monthDf.loc[monthDf[record_status] != 'Whale Moved']
    monthDf = monthDf.loc[monthDf[record_status] != 'Moved']
   
    monthDf_sorted = monthDf.sort_values(by=Subject_Lable)
    monthDf_sorted = monthDf_sorted.groupby(Subject_Lable).apply(lambda x: x.sort_values(by=Subject_Lable)).reset_index(drop=True)

    for scenarioName in ppt_ordered_Scenarios:
        scenarioNamed_rows = monthDf_sorted[monthDf_sorted[Subject_Lable] == scenarioName]
        monthDf_sorted = monthDf_sorted[monthDf_sorted[Subject_Lable] != scenarioName]
        monthDf_sorted = pd.concat([monthDf_sorted, scenarioNamed_rows], ignore_index=True)
    monthDf_sorted['Source'] = file
    monthsDfs.append(monthDf_sorted)
    
    scenario_column = monthDf_sorted[Scenario_Lable].unique().tolist()
    ls_allMainDf = ls_allMainDf + scenario_column
    
    subject_colum = monthDf_sorted[Subject_Lable].unique().tolist()
    ls_allSubDf = ls_allSubDf + subject_colum
    

ls_allMainDf = list(set(ls_allMainDf))
ls_allSubDf = list(set(ls_allSubDf))
print("All Main Scenario: ", ls_allMainDf)
print("All Sub Scenario: ", ls_allSubDf)

df_scenarios = pd.DataFrame()
df_subjects = pd.DataFrame()

print(monthsDfs)

for dataM in monthsDfs:
    source_name = dataM['Source'].iloc[0].split('_')[1].split('.')[0]
    source_CCID_Name = source_name + "_CCID"
    source_CU_Name = source_name + "_CU"
    for main_value in ls_allMainDf:
        # 得到总数
        sub_rows_cur = dataM[dataM[Scenario_Lable] == main_value]
        main_cu_cur = sub_rows_cur['Current CU'].sum()
        main_ccid_cur = len(sub_rows_cur)
        
        # 写入df_scenarios相应的第一列为main_value的行的第source_name列
        df_scenarios.loc[main_value, source_CCID_Name] = main_ccid_cur
        df_scenarios.loc[main_value, source_CU_Name] = main_cu_cur
        
    for sub_value in ls_allSubDf:
        sub_rows_cur = dataM[dataM[Subject_Lable] == sub_value]
        sub_cu_cur = sub_rows_cur['Current CU'].sum()
        sub_ccid_cur = len(sub_rows_cur)
        
        df_subjects.loc[sub_value, source_CCID_Name] = sub_ccid_cur
        df_subjects.loc[sub_value, source_CU_Name] = sub_cu_cur
print("---------------------------------")
print(tabulate(df_scenarios, headers='keys', tablefmt='psql'))
print("---------------------------------")
print(tabulate(df_subjects, headers='keys', tablefmt='psql'))
print("---------------------------------")

df_scenarios.to_excel('Trend_Scenario.xlsx', index=True, header=True)
df_subjects.to_excel('Trend_Subject.xlsx', index=True, header=True)
