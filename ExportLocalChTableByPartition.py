import clickhouse_driver
import pandas as pd
from datetime import datetime

ch_host = 'localhost'
ch_port = 9000
ch_database = 'titan_byod_prod'
ch_table = 'TravelHubLogs_local'

output_dir = '/mnt/d/temp'

connection = clickhouse_driver.connect(host=ch_host, port=ch_port, database=ch_database)

query = f"SELECT * FROM {ch_table} WHERE toDate(EventInfo_DateTime) = '2023-05-17'"
cursor = connection.cursor()
cursor.execute(query)

#date_str = datetime.now().strftime("%Y-%m-%d")
date_str = '2023-05-17'
output_file = f"{output_dir}/{date_str}.parquet"
data = cursor.fetchall()
df = pd.DataFrame(data, columns=[column[0] for column in cursor.description])
df.to_parquet(output_file, index=False)

cursor.close()
connection.disconnect()
