import mysql.connector

# 建立数据库连接
conn = mysql.connector.connect(
    host="titanbyod.mysql.database.azure.com",
    user="TitanAdmin@titanbyod",
    password="dm*QAZ/Titan#2021",
    database="byod_bcp"
)

# 创建一个游标对象
cursor = conn.cursor()

# 打开 TTLDays.txt 文件并按行遍历
with open("TTLDays.txt", "r") as file:
    for line in file:
        # 分割每行的内容
        parts = line.strip().split("\t")

        # 获取 table_name 和 ttl_in_days
        table_name = parts[0]
        ttl_in_days = int(parts[1])

        # 查询 bcp.dataset_v3 表，检查是否存在该 table_name
        query = "SELECT id FROM byod_bcp.dataset_v3 WHERE table_name = %s"
        cursor.execute(query, (table_name,))
        result = cursor.fetchone()

        if result:
            # 如果存在，获取 id 并更新 ttl_in_days
            dataset_id = result[0]
            update_query = "UPDATE byod_bcp.dataset_v3 SET ttl_in_days = %s WHERE id = %s"
            cursor.execute(update_query, (ttl_in_days, dataset_id))
            conn.commit()
            print(f"Updated ttl_in_days {ttl_in_days} for {table_name} (ID: {dataset_id})")

# 关闭游标和连接
cursor.close()
conn.close()
