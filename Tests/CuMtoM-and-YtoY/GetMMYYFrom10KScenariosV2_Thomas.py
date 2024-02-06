import pandas as pd

def process_excel(file_path):
    # 读取 Excel 文件中的 "10K Scenarios" 工作表
    df = pd.read_excel(file_path, sheet_name='10K Scenarios')

    # 替换空值为 "(blank)"
    df['Scenario_w/o Searched'].fillna("(blank)", inplace=True)
    df['Sub Scenario_w/o Searched'].fillna("(blank)", inplace=True)

    # 移除 "Current Status for 10K" 列中值为 "Whale Moved" 的行
    df = df[df['Current Status for 10K'] != 'Whale Moved']

    # 统计结果
    total_rows = len(df)
    total_current_cu = df['Current CU'].sum()

    # Scenario_w/o Searched 统计
    scenario_w_o_searched_counts = df['Scenario_w/o Searched'].value_counts()
    scenario_w_o_searched_sum_cu = df.groupby('Scenario_w/o Searched')['Current CU'].sum()

    # Sub Scenario_w/o Searched 统计
    sub_scenario_counts = df.groupby(['Scenario_w/o Searched', 'Sub Scenario_w/o Searched']).size()
    sub_scenario_sum_cu = df.groupby(['Scenario_w/o Searched', 'Sub Scenario_w/o Searched'])['Current CU'].sum()

    return {
        'total_rows': total_rows,
        'total_current_cu': total_current_cu,
        'scenario_w_o_searched_counts': scenario_w_o_searched_counts,
        'scenario_w_o_searched_sum_cu': scenario_w_o_searched_sum_cu,
        'sub_scenario_counts': sub_scenario_counts,
        'sub_scenario_sum_cu': sub_scenario_sum_cu
    }

# 文件路径
file_paths = ['2023-12.xlsx']

# 处理每个 Excel 文件
for file_path in file_paths:
    result = process_excel(file_path)
    print(f'Results for {file_path}:')
    print(f'Total Rows: {result["total_rows"]}')
    print(f'Total Current CU: {result["total_current_cu"]}')
    print('\nScenario_w/o Searched Counts:')
    print(result['scenario_w_o_searched_counts'])
    print('\nScenario_w/o Searched Sum CU:')
    print(result['scenario_w_o_searched_sum_cu'])
    print('\nSub Scenario Counts:')
    print(result['sub_scenario_counts'])
    print('\nSub Scenario Sum CU:')
    print(result['sub_scenario_sum_cu'])
    print('\n' + '='*50 + '\n')
