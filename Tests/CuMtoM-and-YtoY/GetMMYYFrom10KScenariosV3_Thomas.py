import pandas as pd
import re
from tabulate import tabulate

def clear_characters(text):
    return re.sub('\W', '', text)

def calculate_ratio(num1, num2):
    # 检查 num1 是否为空，如果为空，则设为0
    num1 = 0 if num1 is None else num1
    
    # 检查 num2 是否为0或为空
    if num2 is None or num2 == 0:
        return "#DIV/0!"
    
    # 执行除法运算并格式化结果为小数点后4位
    result = round((num1 - num2) / num2, 4)
    return result

# 参数行，根据具体文件结构进行修改

# 读取当前月份文件2023-12.xlsx的 sheet '10K Scenarios' 到dfCur， status 列是'Current Status for 10K'
file_path_Cur = '2023-12.xlsx'
sheet_name_Cur = '10K Scenarios'
record_status_Cur = 'Current Status for 10K'

# 读取上个月份文件2023-11.xlsx的 sheet '10K Scenarios' 到dfMM， status 列是'Current Status'
file_path_MM = '2023-11.xlsx'
sheet_name_MM = '10K Scenarios'
record_status_MM = 'Current Status'

# 读取去年同月份文件2022-11.xlsx的 sheet '10K Scenarios' 到dfYY， status 列是'Current Status'
file_path_YY = '2022-12.xlsx'
sheet_name_YY = '10K Scenarios'
record_status_YY = 'Current Status'


dfYY = pd.read_excel(file_path_YY, sheet_name=sheet_name_YY)
# 替换dfYY中'Scenario_w/o Searched'列和'Sub Scenario_w/o Searched'列值为0的为"(blank)"
dfYY['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfYY['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfYY['Scenario_w/o Searched'] = dfYY['Scenario_w/o Searched'].apply(clear_characters)
dfYY['Sub Scenario_w/o Searched'] = dfYY['Sub Scenario_w/o Searched'].apply(clear_characters)

dfMM = pd.read_excel(file_path_MM, sheet_name=sheet_name_MM)
# 替换dfYY中'Scenario_w/o Searched'列和'Sub Scenario_w/o Searched'列值为0的为"(blank)"
dfMM['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfMM['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfMM['Scenario_w/o Searched'] = dfMM['Scenario_w/o Searched'].apply(clear_characters)
dfMM['Sub Scenario_w/o Searched'] = dfMM['Sub Scenario_w/o Searched'].apply(clear_characters)

dfCur = pd.read_excel(file_path_Cur, sheet_name=sheet_name_Cur)
# 替换dfYY中'Scenario_w/o Searched'列和'Sub Scenario_w/o Searched'列值为0的为"(blank)"
dfCur['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfCur['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfCur['Scenario_w/o Searched'] = dfCur['Scenario_w/o Searched'].apply(clear_characters)
dfCur['Sub Scenario_w/o Searched'] = dfCur['Sub Scenario_w/o Searched'].apply(clear_characters)


# 去掉dfYY中"Current Status for 10K"列值为"Whale Moved"的行
dfYY = dfYY.loc[dfYY[record_status_YY] != 'Whale Moved']
# 去掉dfMM中"Current Status for 10K"列值为"Whale Moved"的行
dfMM = dfMM.loc[dfMM[record_status_MM] != 'Whale Moved']
# 去掉dfCur中"Current Status for 10K"列值为"Whale Moved"的行
dfCur = dfCur.loc[dfCur[record_status_Cur] != 'Whale Moved']
# 去掉dfYY中"Current Status for 10K"列值为"Moved"的行
dfYY = dfYY.loc[dfYY[record_status_YY] != 'Moved']
# 去掉dfMM中"Current Status for 10K"列值为"Moved"的行
dfMM = dfMM.loc[dfMM[record_status_MM] != 'Moved']
# 去掉dfCur中"Current Status for 10K"列值为"Moved"的行
dfCur = dfCur.loc[dfCur[record_status_Cur] != 'Moved']

# 提取 "Scenario_w/o Searched" 列并合并
scenario_column_cur = dfCur['Scenario_w/o Searched']
scenario_column_mm = dfMM['Scenario_w/o Searched']
scenario_column_yy = dfYY['Scenario_w/o Searched']

# 合并并去重
allMainls = pd.concat([scenario_column_cur, scenario_column_mm, scenario_column_yy], ignore_index=True).unique().tolist()

dictMainToSubs = {}

for main_value in allMainls:
    # 从每个数据框中获取对应主值的子值列表
    subs_cur = dfCur.loc[dfCur['Scenario_w/o Searched'] == main_value, 'Sub Scenario_w/o Searched'].unique().tolist()
    subs_mm = dfMM.loc[dfMM['Scenario_w/o Searched'] == main_value, 'Sub Scenario_w/o Searched'].unique().tolist()
    subs_yy = dfYY.loc[dfYY['Scenario_w/o Searched'] == main_value, 'Sub Scenario_w/o Searched'].unique().tolist()

    # 合并并去重子值列表
    merged_subs = list(set(subs_cur + subs_mm + subs_yy))

    # 存入字典
    dictMainToSubs[main_value] = merged_subs


#print(allMainls)
#print(dictMainToSubs)

print("--------------")
CU_Cur_Total=dfCur['Current CU'].sum()
Len_Cur_Total=len(dfCur)
MMCU=dfMM['Current CU'].sum()
MMLen=len(dfMM)
YYCU=dfYY['Current CU'].sum()
YYLen=len(dfYY)
# perMMCU=(CU_Cur_Total-MMCU)/MMCU
# perMMLen=(Len_Cur_Total-MMLen)/MMLen
# perYYCU=(CU_Cur_Total-YYCU)/YYCU
# perYYLen=(Len_Cur_Total-YYLen)/YYLen

perMMCU=calculate_ratio(CU_Cur_Total, MMCU)
perMMLen=calculate_ratio(Len_Cur_Total, MMLen)
perYYCU=calculate_ratio(CU_Cur_Total, YYCU)
perYYLen=calculate_ratio(Len_Cur_Total, YYLen)
print(f"{file_path_YY}\nCU_Cur_Total\t{CU_Cur_Total}\tLen_Cur_Total\t{Len_Cur_Total}\tCU_MM_Total\t{perMMCU:.2%}\tLen_MM_Total\t{perMMLen:.2%}\tCU_YY_Total\t{perYYCU:.2%}\tLen_YY_Total\t{perYYLen:.2%}")

df = pd.DataFrame(columns=["Name", "Current CU", "Current CCIDs", "MoM (CU)", "MoM (CCIDs)", "MoM% (CU)", "MoM% (CCIDs)", "YoY (CU)", "YoY (CCIDs)", "YoY% (CU)", "YoY% (CCIDs)"])

for main_value in allMainls:
    # Current main_value
    main_rows_cur = dfCur[dfCur['Scenario_w/o Searched'] == main_value]
    main_num_cur = len(main_rows_cur)
    main_cu_cur = main_rows_cur['Current CU'].sum()
    
    # MM main_value
    main_rows_MM = dfMM[dfMM['Scenario_w/o Searched'] == main_value]
    main_num_MM = len(main_rows_MM)
    main_cu_MM = main_rows_MM['Current CU'].sum()
    
    # YY main_value
    main_rows_YY = dfYY[dfYY['Scenario_w/o Searched'] == main_value]
    main_num_YY = len(main_rows_YY)
    main_cu_YY = main_rows_YY['Current CU'].sum()
    
    # Calc
    #main_perMMCU=(main_cu_cur-main_cu_MM)/main_cu_MM
    #main_perMMLen=(main_num_cur-main_num_MM)/main_num_MM
    #main_perYYCU=(main_cu_cur-main_cu_YY)/main_cu_YY
    #main_perYYLen=(main_num_cur-main_num_YY)/main_num_YY
    main_perMMCU=calculate_ratio(main_cu_cur, main_cu_MM)
    main_perMMLen=calculate_ratio(main_num_cur, main_num_MM)
    main_perYYCU=calculate_ratio(main_cu_cur, main_cu_YY)
    main_perYYLen=calculate_ratio(main_num_cur, main_num_YY)
    
    
    new_row_data = {"Name": main_value, "Current CU": main_cu_cur, "Current CCIDs": main_num_cur, "MoM (CU)": main_cu_MM, "MoM (CCIDs)": main_num_MM,  "MoM% (CU)": main_perMMCU, "MoM% (CCIDs)": main_perMMLen, "YoY (CU)": main_cu_YY, "YoY (CCIDs)": main_num_YY, "YoY% (CU)": main_perYYCU, "YoY% (CCIDs)": main_perYYLen}
    df.loc[len(df)] = new_row_data
    
    # 打印结果
    print(f"{main_value}\tSum_CU\t{main_cu_cur}\tSum_Count\t{main_num_cur}")
    subs=dictMainToSubs.get(main_value, [])
    for sub_value in subs:
        # Current sub_value
        sub_rows_cur = dfCur[(dfCur['Scenario_w/o Searched'] == main_value) & (dfCur['Sub Scenario_w/o Searched'] == sub_value)]
        sub_num_cur = len(sub_rows_cur)
        sub_cu_cur = sub_rows_cur['Current CU'].sum()
        
        # MM sub_value
        sub_rows_MM = dfMM[(dfMM['Scenario_w/o Searched'] == main_value) & (dfMM['Sub Scenario_w/o Searched'] == sub_value)]
        sub_num_MM = len(sub_rows_MM)
        sub_cu_MM = sub_rows_MM['Current CU'].sum()
        
        # YY sub_value
        sub_rows_YY = dfYY[(dfYY['Scenario_w/o Searched'] == main_value) & (dfYY['Sub Scenario_w/o Searched'] == sub_value)]
        sub_num_YY = len(sub_rows_YY)
        sub_cu_YY = sub_rows_YY['Current CU'].sum()
        
        # Calc
        #sub_perMMCU=(sub_cu_cur-sub_cu_MM)/sub_cu_MM
        #sub_perMMLen=(sub_num_cur-sub_num_MM)/sub_num_MM
        #sub_perYYCU=(sub_cu_cur-sub_cu_YY)/sub_cu_YY
        #sub_perYYLen=(sub_num_cur-sub_num_YY)/sub_num_YY
        sub_perMMCU=calculate_ratio(sub_cu_cur, sub_cu_MM)
        sub_perMMLen=calculate_ratio(sub_num_cur, sub_num_MM)
        sub_perYYCU=calculate_ratio(sub_cu_cur, sub_cu_YY)
        sub_perYYLen=calculate_ratio(sub_num_cur, sub_num_YY)
    
        new_row_data = {"Name": "-- " + sub_value, "Current CU": sub_cu_cur, "Current CCIDs": sub_num_cur, "MoM (CU)": sub_cu_MM, "MoM (CCIDs)": sub_num_MM,  "MoM% (CU)": sub_perMMCU, "MoM% (CCIDs)": sub_perMMLen, "YoY (CU)": sub_cu_YY, "YoY (CCIDs)": sub_num_YY, "YoY% (CU)": sub_perYYCU, "YoY% (CCIDs)": sub_perYYLen}
        df.loc[len(df)] = new_row_data
        
        print(f"    {sub_value}")
        # 打印结果
        print(f"            Sub Number of rows:     {sub_num_cur}")
        print(f"            Sub Total 'Current CU': {sub_cu_cur}")

    print()
    print("......")
    
new_row_data = {"Name": "Grand Total", "Current CU": CU_Cur_Total, "Current CCIDs": Len_Cur_Total, "MoM (CU)": MMCU, "MoM (CCIDs)": MMLen, "MoM% (CU)": perMMCU, "MoM% (CCIDs)": perMMLen, "YoY (CU)": YYCU, "YoY (CCIDs)": YYLen, "YoY% (CU)": perYYCU, "YoY% (CCIDs)": perYYLen}
df.loc[len(df)] = new_row_data

print(tabulate(df, headers='keys', tablefmt='plain', showindex=False, colalign='left'))

df.to_excel('MMYY_output.xlsx', index=False, header=True)

# 打印完成消息
print("DataFrame 已保存为 output.xlsx")

# 使用.str.strip()去除字符串两端的空格并找到'Scenario_w/o Searched'值为'Caption'的行
#caption_rows = dfCur[dfCur['Scenario_w/o Searched'].str.strip() == 'Caption']

# 打印行的数量
#num_caption_rows = len(caption_rows)
#print(f"Number of rows with 'Scenario_w/o Searched' equal to 'Caption': {num_caption_rows}")

# 如果需要，打印这些行的“Current CU”列的总数
#total_current_cu = caption_rows['Current CU'].sum()
#print(f"Total 'Current CU' for rows with 'Scenario_w/o Searched' equal to 'Caption': {total_current_cu}")

#print("--------------")
#for main_value, subs_values in dictMainToSubs.items():
#    print(f"For {main_value}:")
#    print(f"  Key: {main_value}")
#    for sub_value in subs_values:
#        print(f"  Value: {sub_value}")
#    print()

