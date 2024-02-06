import pandas as pd
from openpyxl import load_workbook
import sys

def main():
    # 获取命令行参数

    file_name="Speech Verbatims -2023.xlsx"
    currentM="Dec'23"
    lastM="Nov'23"
    sumPage="MtoM-Summary"
    
    current_df = pd.read_excel(file_name, sheet_name=currentM)
    last_df = pd.read_excel(file_name, sheet_name=lastM)

    # 统计P列和Q列的元素出现次数
    currentDSAT = dict(current_df['DSAT'].value_counts())
    currentSAT = dict(current_df['SAT'].value_counts())
    lastDSAT = dict(last_df['DSAT'].value_counts())
    lastSAT = dict(last_df['SAT'].value_counts())

    # 获取去重后的key并集
    keyDSAT = list(set(currentDSAT.keys()).union(set(lastDSAT.keys())))
    keySAT = list(set(currentSAT.keys()).union(set(lastSAT.keys())))

    retCurCol=str(currentM).replace("'", "-")
    lastCurCol=str(lastM).replace("'", "-")
    
    # DSAT
    
    # 创建结果DSAT DataFrame
    resultDSAT = pd.DataFrame(columns=['Key', str(currentM), str(lastM), str(retCurCol), str(lastCurCol)])

    # 填充结果DataFrame
    resultDSAT['Key'] = keyDSAT
    resultDSAT[str(currentM)] = resultDSAT['Key'].map(currentDSAT).fillna(0).astype(int)
    resultDSAT[str(lastM)] = resultDSAT['Key'].map(lastDSAT).fillna(0).astype(int)
    total_DSAT_currentM = resultDSAT[str(currentM)].sum()
    total_DSAT_lastM = resultDSAT[str(lastM)].sum()
    
    #resultDSAT[str(retCurCol)] = resultDSAT.apply(lambda row: f"{(row[str(currentM)] / total_DSAT_currentM * 100):.0f}%", axis=1)
    #resultDSAT[str(lastCurCol)] = resultDSAT.apply(lambda row: f"{(row[str(lastM)] / total_DSAT_lastM * 100):.0f}%", axis=1)
    resultDSAT[str(retCurCol)] = resultDSAT.apply(lambda row: round((row[str(currentM)] / total_DSAT_currentM), 2), axis=1)
    resultDSAT[str(lastCurCol)] = resultDSAT.apply(lambda row: round((row[str(lastM)] / total_DSAT_lastM), 2), axis=1)
    
    # 按照 str(currentM) 列的值从大到小排序
    resultDSAT.sort_values(by=[str(currentM), 'Key'], ascending=[False, True], inplace=True)

    #print(f"total_DSAT_currentM {total_DSAT_currentM}")
    #print(f"total_DSAT_lastM    {total_DSAT_lastM}")
    
    new_row_data = pd.Series(["DSAT", total_DSAT_currentM, total_DSAT_lastM, "100%", "100%"], index=resultDSAT.columns)
    #resultDSAT.loc[-1] = new_row_data
    #resultDSAT.index = resultDSAT.index + 1  # 重新调整索引，使新行成为第一行
    #resultDSAT = resultDSAT.sort_index()    # 按索引排序
    resultDSAT.loc[len(resultDSAT)] = new_row_data
    
    while len(resultDSAT)<19 :
        new_row = pd.Series([None, None, None, None, None], index=resultDSAT.columns)
        resultDSAT.loc[len(resultDSAT)] = new_row
    
    
    
    # SAT
    
    # 创建结果SAT DataFrame
    resultSAT = pd.DataFrame(columns=['Key', str(currentM), str(lastM), str(retCurCol), str(lastCurCol)])

    # 填充结果DataFrame
    resultSAT['Key'] = keySAT
    resultSAT[str(currentM)] = resultSAT['Key'].map(currentSAT).fillna(0).astype(int)
    resultSAT[str(lastM)] = resultSAT['Key'].map(lastSAT).fillna(0).astype(int)
    total_SAT_currentM = resultSAT[str(currentM)].sum()
    total_SAT_lastM = resultSAT[str(lastM)].sum()
    
    #resultSAT[str(retCurCol)] = resultSAT.apply(lambda row: f"{(row[str(currentM)] / total_SAT_currentM * 100):.0f}%", axis=1)
    #resultSAT[str(lastCurCol)] = resultSAT.apply(lambda row: f"{(row[str(lastM)] / total_SAT_lastM * 100):.0f}%", axis=1)
    resultSAT[str(retCurCol)] = resultSAT.apply(lambda row: round((row[str(currentM)] / total_SAT_currentM), 2), axis=1)
    resultSAT[str(lastCurCol)] = resultSAT.apply(lambda row: round((row[str(lastM)] / total_SAT_lastM), 2), axis=1)
    
    # 按照 str(currentM) 列的值从大到小排序
    resultSAT.sort_values(by=[str(currentM), 'Key'], ascending=[False, True], inplace=True)

    #print(f"total_SAT_currentM {total_SAT_currentM}")
    #print(f"total_SAT_lastM    {total_SAT_lastM}")
    
    new_row_data = pd.Series(["SAT", total_SAT_currentM, total_SAT_lastM, "100%", "100%"], index=resultSAT.columns)
    #resultSAT.loc[-1] = new_row_data
    #resultSAT.index = resultSAT.index + 1  # 重新调整索引，使新行成为第一行
    #resultSAT = resultSAT.sort_index()    # 按索引排序
    resultSAT.loc[len(resultSAT)] = new_row_data
    
    while len(resultSAT)<19 :
        new_row = pd.Series([None, None, None, None, None], index=resultSAT.columns)
        resultSAT.loc[len(resultSAT)] = new_row


    # RESULT
    
    result_df = pd.concat([resultDSAT, resultSAT], ignore_index=True)    
    result_df.insert(loc=3, column='Item', value=result_df.iloc[:, 0])
    
    print(result_df)
    
    output_file="out_" + file_name
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as excel_writer:
        # 将 result_df 写入指定 sheet
        result_df.to_excel(excel_writer, sheet_name='MtoM-Summary', index=False)
    
        # 获取已有的 workbook
        workbook = excel_writer.book
    
        # 保存修改后的 Excel 文件
        workbook.save(output_file)

if __name__ == "__main__":
    main()
