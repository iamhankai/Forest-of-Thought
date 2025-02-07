import pandas as pd
import jsonlines

# 读取Parquet文件
parquet_file_path = 'aime_2024_problems.parquet'
df = pd.read_parquet(parquet_file_path, engine='pyarrow')  # 或者使用 engine='fastparquet'

# 将DataFrame转换为字典记录列表
records = df.to_dict(orient='records')

# 写入JSON Lines文件
jsonl_file_path = 'aime_2024_problems.jsonl'
with jsonlines.open(jsonl_file_path, mode='w') as writer:
    writer.write_all(records)

print(f"数据已成功从 {parquet_file_path} 转换并写入到 {jsonl_file_path}")