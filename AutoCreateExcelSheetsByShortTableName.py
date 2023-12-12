import openpyxl

# 打开源文件
src_file = "V3-V2-IngestionCompare20230815.xlsx"
workbook = openpyxl.load_workbook(src_file)

# 获取"Tables Count"工作表
tables_count_sheet = workbook["Sheet Names"]

# 遍历B列从B2到最后一个不为空的单元格
for cell in tables_count_sheet["B2:B" + str(tables_count_sheet.max_row)]:
    for row in cell:
        if row.value:
            table_name = row.value
            # 复制"Template"工作表的内容并重命名为table_name
            template_sheet = workbook["Template"]
            new_sheet = workbook.copy_worksheet(template_sheet)
            new_sheet.title = table_name

# 保存修改后的文件
output_file = "Modified_" + src_file
workbook.save(output_file)
print("新的工作表已添加并保存为：", output_file)
