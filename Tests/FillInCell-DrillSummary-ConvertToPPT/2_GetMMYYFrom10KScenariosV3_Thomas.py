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

def GetTop2or3RecordsByCol(sub_subject_records, colName,  isDec):
    # 过滤掉 colName 列内容是字符串的行, 先筛选出含有 int 或 float 数值的行
    filtered_records = [record for record in sub_subject_records if isinstance(record[colName], (int, float))]
    if isDec:
        filtered_subset = [record for record in filtered_records if record[colName] < 0]
        sorted_mumber_records = sorted(filtered_subset, key=lambda x: x[colName], reverse=False)
    else:
        filtered_subset = [record for record in filtered_records if record[colName] > 0]
        sorted_mumber_records = sorted(filtered_subset, key=lambda x: x[colName], reverse=True)
    
    topRecors = []
    if (len(sorted_mumber_records) < 1):
        return topRecors
    elif (len(sorted_mumber_records) < 2):
        return sorted_mumber_records[:1]
    elif (len(sorted_mumber_records) < 3):
        return sorted_mumber_records[:2]
    else:
        topRecors = sorted_mumber_records[:3]
        if abs((topRecors[1][colName] - topRecors[2][colName]) / topRecors[2][colName]) > 0.3:
            return topRecors[:2]
        else:
            return topRecors[:3]



def TagCustomerIncreaseOrDecrease(topMetricItems, dict_SubjectCustomerCU, col, order, targetDf, targetDf_Col):
    # 如果没有数据，直接返回
    if (len(topMetricItems) < 1):
        return
    
    merged_cur_to_MM_Names= ""
    subject_values = [d["Subject"] for d in topMetricItems if "Subject" in d]
    #print("-------------TagCustomerIncreaseOrDecrease-------------")
    for subject in subject_values:
        #print(f"    Subject: {subject}")
        #print(f"    col: {col}")
        #print(f"    order: {order}")
        customers = dict_SubjectCustomerCU[subject]
        customers.to_excel(f"MMYY_{subject}_{col}_{order}_output.xlsx", index=False, header=True)
        #---------------------
        # 找到 MoM_CU 列大于 0 的行
        if order == "increased":
            recordsMoreThanBaseline = customers[customers[col] > 0]
        else:
            recordsMoreThanBaseline = customers[customers[col] < 0]
        
        # 如果行数大于0
        if len(recordsMoreThanBaseline) > 0:
            # 找到 MoM_CU 列中值最大的前2行
            if order == "increased":
                top_metric_rows = recordsMoreThanBaseline.nlargest(2, col)
            else:
                top_metric_rows = recordsMoreThanBaseline.nsmallest(2, col)
        
            top_metric_rows['Name'] = top_metric_rows['Name'].apply(lambda x: x + " (" + order + ")")
        
            # 更新原始 DataFrame
            customers.update(top_metric_rows)
            customers = customers.sort_values(by='Current CU', ascending=False)
            customers_Names = customers['Name'].tolist()
            merged_cur_to_MM_Names = ", ".join(customers_Names)
            targetDf.loc[targetDf['Subject'] == subject, targetDf_Col] = merged_cur_to_MM_Names
        
        #print(merged_cur_to_MM_Names)
        #---------------------
    return

# 参数行，根据具体文件结构进行修改

# 读取当前月份文件2023-12.xlsx的 sheet '10K Scenarios' 到dfCur， status 列是'Current Status for 10K'
file_path_Cur = 'out_2024-3.xlsx'
sheet_name_Cur = 'Sheet1'
record_status_Cur = 'Current Status'

# 读取上个月份文件2023-11.xlsx的 sheet '10K Scenarios' 到dfMM， status 列是'Current Status'
file_path_MM = 'out_2024-2.xlsx'
sheet_name_MM = 'Sheet1'
record_status_MM = 'Current Status'

# 读取去年同月份文件2022-11.xlsx的 sheet '10K Scenarios' 到dfYY， status 列是'Current Status'
file_path_YY = 'out_2023-3.xlsx'
sheet_name_YY = 'Sheet1'
record_status_YY = 'Current Status'

cur_Mon="Mar"

ppt_ordered_Scenarios = ["Caption", "Transcription", "Voice Agent", "Content Reader", "Language Learning", "Speech To Speech Translation", "Blank"]

dfYY = pd.read_excel(file_path_YY, sheet_name=sheet_name_YY)
# 替换dfYY中'Scenario_w/o Searched'列和'Sub Scenario_w/o Searched'列值为0的为"(blank)"
dfYY['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfYY['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfYY['Name'].fillna(dfYY["Cloud Customer GUID"], inplace=True)
dfYY['Current CU'].fillna(0, inplace=True)
dfYY['Scenario_w/o Searched'] = dfYY['Scenario_w/o Searched'].apply(process_column_content) #.str.strip() #.apply(clear_characters)
dfYY['Sub Scenario_w/o Searched'] = dfYY['Sub Scenario_w/o Searched'].apply(process_column_content) #.str.strip() #.apply(clear_characters)

dfMM = pd.read_excel(file_path_MM, sheet_name=sheet_name_MM)
# 替换dfYY中'Scenario_w/o Searched'列和'Sub Scenario_w/o Searched'列值为0的为"(blank)"
dfMM['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfMM['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfMM['Name'].fillna(dfMM["Cloud Customer GUID"], inplace=True)
dfMM['Current CU'].fillna(0, inplace=True)
dfMM['Scenario_w/o Searched'] = dfMM['Scenario_w/o Searched'].apply(process_column_content) #.str.strip() #.apply(clear_characters)
dfMM['Sub Scenario_w/o Searched'] = dfMM['Sub Scenario_w/o Searched'].apply(process_column_content) #.str.strip() #.apply(clear_characters)

dfCur = pd.read_excel(file_path_Cur, sheet_name=sheet_name_Cur)
# 替换dfYY中'Scenario_w/o Searched'列和'Sub Scenario_w/o Searched'列值为0的为"(blank)"
dfCur['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfCur['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)
dfCur['Name'].fillna(dfCur["Cloud Customer GUID"], inplace=True)
dfCur['Current CU'].fillna(0, inplace=True)
dfCur['Scenario_w/o Searched'] = dfCur['Scenario_w/o Searched'].apply(process_column_content) #.str.strip() #.apply(clear_characters)
dfCur['Sub Scenario_w/o Searched'] = dfCur['Sub Scenario_w/o Searched'].apply(process_column_content) #.str.strip() #.apply(clear_characters)


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

perMMCU=calculate_ratio(CU_Cur_Total, MMCU)
perMMLen=calculate_ratio(Len_Cur_Total, MMLen)
perYYCU=calculate_ratio(CU_Cur_Total, YYCU)
perYYLen=calculate_ratio(Len_Cur_Total, YYLen)
#print(f"{file_path_YY}\nCU_Cur_Total\t{CU_Cur_Total}\tLen_Cur_Total\t{Len_Cur_Total}\tCU_MM_Total\t{perMMCU:.2%}\tLen_MM_Total\t{perMMLen:.2%}\tCU_YY_Total\t{perYYCU:.2%}\tLen_YY_Total\t{perYYLen:.2%}")

df = pd.DataFrame(columns=["Scenario", "Subject", "Current CU", "Current CCIDs", "MoM (CU)", "MoM (CCIDs)", "MoM% (CU)", "MoM% (CCIDs)", "YoY (CU)", "YoY (CCIDs)", "YoY% (CU)", "YoY% (CCIDs)", "Names (MoM)", "Names (YoY)"])
sub_subject_Names=[]
sub_subject_records=[]
dictMM_SubjectCustomerCU = {}
dictYY_SubjectCustomerCU = {}
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
    
    # Calc ratio
    main_perMMCU=calculate_ratio(main_cu_cur, main_cu_MM)
    main_perMMLen=calculate_ratio(main_num_cur, main_num_MM)
    main_perYYCU=calculate_ratio(main_cu_cur, main_cu_YY)
    main_perYYLen=calculate_ratio(main_num_cur, main_num_YY)
    
    
    new_row_data = {"Scenario": main_value, "Subject": "$ " + main_value, "Current CU": main_cu_cur, "Current CCIDs": main_num_cur, "MoM (CU)": main_cu_MM, "MoM (CCIDs)": main_num_MM,  "MoM% (CU)": main_perMMCU, "MoM% (CCIDs)": main_perMMLen, "YoY (CU)": main_cu_YY, "YoY (CCIDs)": main_num_YY, "YoY% (CU)": main_perYYCU, "YoY% (CCIDs)": main_perYYLen, "Names (MoM)": "", "Names (YoY)": ""}
    df.loc[len(df)] = new_row_data
    
    # 打印结果
    #print(f"{main_value}\tSum_CU\t{main_cu_cur}\tSum_Count\t{main_num_cur}")
    sub_subject_Names=dictMainToSubs.get(main_value, [])
    for sub_value in sub_subject_Names:
        # Current sub_value
        sub_rows_cur = dfCur[(dfCur['Scenario_w/o Searched'] == main_value) & (dfCur['Sub Scenario_w/o Searched'] == sub_value)]
        #print(type(sub_rows_cur))
        sub_rows_cur['MoM_CU'] = 0
        sub_rows_cur['MoM_CCID'] = 0
        sub_rows_cur['YoY_CU'] = 0
        sub_rows_cur['YoY_CCID'] = 0
        sub_num_cur = len(sub_rows_cur)
        sub_cu_cur = sub_rows_cur['Current CU'].sum()
        
        # MM sub_value
        sub_rows_MM = dfMM[(dfMM['Scenario_w/o Searched'] == main_value) & (dfMM['Sub Scenario_w/o Searched'] == sub_value)]
        sub_rows_MM['MoM_CU'] = 0
        sub_rows_MM['MoM_CCID'] = 0
        sub_rows_MM['YoY_CU'] = 0
        sub_rows_MM['YoY_CCID'] = 0
        sub_num_MM = len(sub_rows_MM)
        sub_cu_MM = sub_rows_MM['Current CU'].sum()
        
        # YY sub_value
        sub_rows_YY = dfYY[(dfYY['Scenario_w/o Searched'] == main_value) & (dfYY['Sub Scenario_w/o Searched'] == sub_value)]
        sub_rows_YY['MoM_CU'] = 0
        sub_rows_YY['YoY_CU'] = 0
        sub_num_YY = len(sub_rows_YY)
        sub_cu_YY = sub_rows_YY['Current CU'].sum()
        
        # Calc ratio
        sub_perMMCU=calculate_ratio(sub_cu_cur, sub_cu_MM)
        sub_perMMLen=calculate_ratio(sub_num_cur, sub_num_MM)
        sub_perYYCU=calculate_ratio(sub_cu_cur, sub_cu_YY)
        sub_perYYLen=calculate_ratio(sub_num_cur, sub_num_YY)
        
        # 得到custome的Name以CU降序排序及标记后的数据
        # MM
        #   New customer
        new_cur_to_MM = sub_rows_cur[~sub_rows_cur['Cloud Customer GUID'].isin(sub_rows_MM['Cloud Customer GUID'])]
        new_cur_to_MM['Name'] = new_cur_to_MM['Name'] + " (" + new_cur_to_MM[record_status_Cur].str.lower() +")"
        #   Retain customer
        retain_cur_to_MM = sub_rows_cur[sub_rows_cur['Cloud Customer GUID'].isin(sub_rows_MM['Cloud Customer GUID'])]
        retain_MM_to_cur = sub_rows_MM[sub_rows_MM['Cloud Customer GUID'].isin(sub_rows_cur['Cloud Customer GUID'])]
        #print(f"Debug retain_cur_to_MM 1 \n {type(retain_cur_to_MM)} {retain_cur_to_MM}")
        #print(f"Debug retain_MM_to_cur 1 \n {type(retain_MM_to_cur)} {retain_MM_to_cur}")
        #print(f"****************{retain_cur_to_MM['Name'].tolist()}")
        #retain_cur_to_MM['MoM_CU'] = round((retain_cur_to_MM['Current CU'] - retain_MM_to_cur['Current CU']) / retain_MM_to_cur['Current CU'], 4)
        for index, record in retain_cur_to_MM.iterrows():
            ccid = record['Cloud Customer GUID']
            try:
                MoM_CU_value = round((record['Current CU'] - retain_MM_to_cur.loc[retain_MM_to_cur['Cloud Customer GUID'] == ccid, 'Current CU'].values[0]) / retain_MM_to_cur.loc[retain_MM_to_cur['Cloud Customer GUID'] == ccid, 'Current CU'].values[0], 4)
            except:
                print(f"record {record}")
                print(f"Error: {ccid} : record {record['Current CU']} , len retain_MM_to_cur {len(retain_MM_to_cur.loc[retain_MM_to_cur['Cloud Customer GUID'] == ccid, 'Current CU'].values)}")
                MoM_CU_value = 0
            retain_cur_to_MM.at[index, 'MoM_CU'] = MoM_CU_value
        #print(f"Debug retain_cur_to_MM 2 \n {retain_cur_to_MM}")
        #   Moved customer
        moved_MM_to_cur = sub_rows_MM[~sub_rows_MM['Cloud Customer GUID'].isin(sub_rows_cur['Cloud Customer GUID'])]
        moved_MM_to_cur['Name'] = moved_MM_to_cur['Name'] + " (moved)"
        moved_MM_to_cur['Current CU'] = 0
        #   Get cur to MM name list
        exist_cur_to_MM = pd.concat([new_cur_to_MM, retain_cur_to_MM], ignore_index=True)
        exist_cur_to_MM = exist_cur_to_MM.sort_values(by='Current CU', ascending=False)
        whole_cur_to_MM = pd.concat([exist_cur_to_MM, moved_MM_to_cur], ignore_index=True)
        whole_cur_to_MM['MoM_CU'].fillna(0, inplace=True)
        whole_cur_to_MM_Names = whole_cur_to_MM['Name'].tolist()
        whole_cur_to_MM_Names = [str(item) for item in whole_cur_to_MM_Names]
        merged_cur_to_MM_Names = ", ".join(whole_cur_to_MM_Names)
        # YY
        #   New customer
        new_cur_to_YY = sub_rows_cur[~sub_rows_cur['Cloud Customer GUID'].isin(sub_rows_YY['Cloud Customer GUID'])]
        new_cur_to_YY['Name'] = new_cur_to_YY['Name'] + " (new)"
        #   Retain customer
        retain_cur_to_YY = sub_rows_cur[sub_rows_cur['Cloud Customer GUID'].isin(sub_rows_YY['Cloud Customer GUID'])]
        retain_YY_to_cur = sub_rows_YY[sub_rows_YY['Cloud Customer GUID'].isin(sub_rows_cur['Cloud Customer GUID'])]
        #print(f"Debug retain_cur_to_YY 1 \n {retain_cur_to_YY}")
        #print(f"Debug retain_YY_to_cur 1 \n {retain_YY_to_cur}")
        #print(f"****************{retain_cur_to_YY['Name'].tolist()}")
        #retain_cur_to_YY['MoM_CU'] = round((retain_cur_to_YY['Current CU'] - retain_YY_to_cur['Current CU']) / retain_YY_to_cur['Current CU'], 4)
        for index, record in retain_cur_to_YY.iterrows():
            ccid = record['Cloud Customer GUID']
            MoM_CU_value = round((record['Current CU'] - retain_YY_to_cur.loc[retain_YY_to_cur['Cloud Customer GUID'] == ccid, 'Current CU'].values[0]) / retain_YY_to_cur.loc[retain_YY_to_cur['Cloud Customer GUID'] == ccid, 'Current CU'].values[0], 4)
            retain_cur_to_YY.at[index, 'YoY_CU'] = MoM_CU_value
        #print(f"Debug retain_cur_to_YY 2 \n {retain_cur_to_YY}")
        #   Moved customer
        moved_YY_to_cur = sub_rows_YY[~sub_rows_YY['Cloud Customer GUID'].isin(sub_rows_cur['Cloud Customer GUID'])]
        moved_YY_to_cur['Name'] = moved_YY_to_cur['Name'] + " (moved)"
        moved_YY_to_cur['Current CU'] = 0
        #   Get cur to YY name list
        exist_cur_to_YY = pd.concat([new_cur_to_YY, retain_cur_to_YY], ignore_index=True)
        exist_cur_to_YY = exist_cur_to_YY.sort_values(by='Current CU', ascending=False)
        whole_cur_to_YY = pd.concat([exist_cur_to_YY, moved_YY_to_cur], ignore_index=True)
        whole_cur_to_YY['YoY_CU'].fillna(0, inplace=True)
        whole_cur_to_YY_Names = whole_cur_to_YY['Name'].tolist()
        whole_cur_to_YY_Names = [str(item) for item in whole_cur_to_YY_Names]
        merged_cur_to_YY_Names = ", ".join(whole_cur_to_YY_Names)
        
        # 存入字典
        #print(f"Debug: {whole_cur_to_MM.columns}")
        #dictMM_SubjectCustomerCU["-- " + sub_value] = whole_cur_to_MM
        #dictYY_SubjectCustomerCU["-- " + sub_value] = whole_cur_to_YY
        dictMM_SubjectCustomerCU[sub_value] = whole_cur_to_MM
        dictYY_SubjectCustomerCU[sub_value] = whole_cur_to_YY
    
        new_row_data = {"Scenario": main_value, "Subject": sub_value, "Current CU": sub_cu_cur, "Current CCIDs": sub_num_cur, "MoM (CU)": sub_cu_MM, "MoM (CCIDs)": sub_num_MM,  "MoM% (CU)": sub_perMMCU, "MoM% (CCIDs)": sub_perMMLen, "YoY (CU)": sub_cu_YY, "YoY (CCIDs)": sub_num_YY, "YoY% (CU)": sub_perYYCU, "YoY% (CCIDs)": sub_perYYLen, "Names (MoM)": merged_cur_to_MM_Names, "Names (YoY)": merged_cur_to_YY_Names}
        sub_subject_records.append(new_row_data)
        #new_row_data["Subject"] = "-- " + new_row_data["Subject"]  # Add symbols
        df.loc[len(df)] = new_row_data
        
        print(f"    {sub_value}")
        # 打印结果
        #print(f"            Sub Number of rows:     {sub_num_cur}")
        #print(f"            Sub Total 'Current CU': {sub_cu_cur}")
        #print(f"            Sub MoM 'merged_cur_to_MM_Names':   {merged_cur_to_MM_Names}")
        #print(f"            Sub YoY 'merged_cur_to_YY_Names':   {merged_cur_to_YY_Names}")

    print()
    print("......")

#df.to_excel('MMYY_output_debug.xlsx', index=False, header=True)

df_sorted = df.sort_values(by='Scenario')
df_sorted = df_sorted.groupby('Scenario').apply(lambda x: x.sort_values(by='Subject')).reset_index(drop=True)
df = df_sorted

for scenarioName in ppt_ordered_Scenarios:
    scenarioNamed_rows = df[df['Scenario'] == scenarioName]
    df = df[df['Scenario'] != scenarioName]
    df = pd.concat([df, scenarioNamed_rows], ignore_index=True)

#blank_rows = df[df['Scenario'] == 'Blank']
#df = df[df['Scenario'] != 'Blank']
#df = pd.concat([df, blank_rows], ignore_index=True)
    
new_row_data = {"Scenario": "Grand Total", "Subject": "Grand Total", "Current CU": CU_Cur_Total, "Current CCIDs": Len_Cur_Total, "MoM (CU)": MMCU, "MoM (CCIDs)": MMLen, "MoM% (CU)": perMMCU, "MoM% (CCIDs)": perMMLen, "YoY (CU)": YYCU, "YoY (CCIDs)": YYLen, "YoY% (CU)": perYYCU, "YoY% (CCIDs)": perYYLen}
df.loc[len(df)] = new_row_data

# 添加 Name 标签 "Increase" 或 "Decrease"
# 按照从大到小排序 sub_subject_records 列表中的字典元素
#sorted_MoM_CCID_sub_subject_records = sorted(sub_subject_records, key=lambda x: x["MoM% (CCIDs)"], reverse=True)
#sorted_MoM_CU_sub_subject_records = sorted(sub_subject_records, key=lambda x: x["MoM% (CU)"], reverse=True)
#sorted_YoY_CCID_sub_subject_records = sorted(sub_subject_records, key=lambda x: x["YoY% (CCIDs)"], reverse=True)
#sorted_YoY_CU_sub_subject_records = sorted(sub_subject_records, key=lambda x: x["YoY% (CU)"], reverse=True)

#topMoMCCID_DecreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "MoM% (CCIDs)", True)
#topMoMCCID_IncreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "MoM% (CCIDs)", False)
#topYoYCCID_DecreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "YoY% (CCIDs)", True)
#topYoYCCID_IncreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "YoY% (CCIDs)", False)


topMoMCU_DecreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "MoM% (CU)", True)
topMoMCU_IncreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "MoM% (CU)", False)
topYoYCU_DecreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "YoY% (CU)", True)
topYoYCU_IncreaseSubjects = GetTop2or3RecordsByCol(sub_subject_records, "YoY% (CU)", False)

    # topMetricItems = topMoMCCID_DecreaseSubjects
    # dict_SubjectCustomerCU = dictMM_SubjectCustomerCU
    # col = "MoM_CU"
    # order = "increased"
TagCustomerIncreaseOrDecrease(topMoMCU_DecreaseSubjects, dictMM_SubjectCustomerCU, "MoM_CU", "decreased", df, "Names (MoM)")
TagCustomerIncreaseOrDecrease(topMoMCU_IncreaseSubjects, dictMM_SubjectCustomerCU, "MoM_CU", "increased", df, "Names (MoM)")
TagCustomerIncreaseOrDecrease(topYoYCU_DecreaseSubjects, dictYY_SubjectCustomerCU, "YoY_CU", "decreased", df, "Names (YoY)")
TagCustomerIncreaseOrDecrease(topYoYCU_IncreaseSubjects, dictYY_SubjectCustomerCU, "YoY_CU", "increased", df, "Names (YoY)")

print(tabulate(df.iloc[:, :-2], headers='keys', tablefmt='plain', showindex=False, colalign='left'))

df.to_excel('MMYY_output.xlsx', index=False, header=True)

def format_currency(x):
    return "${:,.0f}".format(x)

# 自定义函数将百分比转换为指定格式的字符串
def format_percentage(x):
    # 检查单元格是否为浮点数类型，如果是，则执行转换，否则返回原始值
    if isinstance(x, float):
        return "{:.2f}%".format(x * 100)
    else:
        return x

dfppt = df[~df['Subject'].str.startswith('$')]
dfppt = dfppt.iloc[:-2]
ppt_cu_sum = dfppt["Current CU"].sum()
# 计算 "10K CU %" 列的值， 将数据类型设置为 float
dfppt['10K CU %'] = (dfppt['Current CU'] / ppt_cu_sum ).round(4)
dfppt['10K CU %'] = dfppt['10K CU %'].astype(float)
# 将 'Current CU' 列的值舍弃所有小数部分
dfppt['Current CU'] = dfppt['Current CU'].astype(int)

# 应用自定义函数将百分比列转换为字符串列
dfppt['10K CU %'] = dfppt['10K CU %'].apply(format_percentage)
dfppt['MoM% (CU)'] = dfppt['MoM% (CU)'].apply(format_percentage)
dfppt['MoM% (CCIDs)'] = dfppt['MoM% (CCIDs)'].apply(format_percentage)
dfppt['YoY% (CU)'] = dfppt['YoY% (CU)'].apply(format_percentage)
dfppt['YoY% (CCIDs)'] = dfppt['YoY% (CCIDs)'].apply(format_percentage)

# 应用自定义函数将整数列转换为字符串列
dfppt['Current CU'] = dfppt['Current CU'].apply(format_currency)

dfppt = dfppt.rename(columns={'Subject': 'Sub Scenario', 'Current CU': '10 CU @ ' + cur_Mon, 'Current CCIDs': '10K CCID @ ' + cur_Mon})
dfppt = dfppt.rename(columns={'Names (MoM)': 'Top Usage Customer Name​ (ranked by CU) MoM', 'Names (YoY)': 'Top Usage Customer Name​ (ranked by CU) YoY'})

pptmm_cols = ["Scenario", "Sub Scenario", "10 CU @ " + cur_Mon, "10K CU %", "10K CCID @ " + cur_Mon, "Top Usage Customer Name​ (ranked by CU) MoM", "MoM% (CU)", "MoM% (CCIDs)"]
pptyy_cols = ["Scenario", "Sub Scenario", "10 CU @ " + cur_Mon, "10K CU %", "10K CCID @ " + cur_Mon, "Top Usage Customer Name​ (ranked by CU) YoY", "YoY% (CU)", "YoY% (CCIDs)"]

dfppt.to_excel('MMYY_output_ppt.xlsx', index=False, header=True)

dfpptmm = dfppt[pptmm_cols]
dfpptyy= dfppt[pptyy_cols]


dfpptmm.to_excel('MMYY_output_SrcPPT_MoM.xlsx', index=False, header=True)
dfpptyy.to_excel('MMYY_output_SrcPPT_YoY.xlsx', index=False, header=True)

# 打印完成消息
print("DataFrame 已保存为 MMYY_output.xlsx")

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
