import pandas as pd
from tqdm import tqdm

# 路径设置
label_path = r'G:\咸海流域\24\BE\times_series\label_1.csv'
band_paths = [
    r'G:\咸海流域\24\BE\times_series\Band_1.csv',
    r'G:\咸海流域\24\BE\times_series\Band_2.csv',
    r'G:\咸海流域\24\BE\times_series\Band_3.csv',
    r'G:\咸海流域\24\BE\times_series\Band_4.csv',
]

output_paths = [
    r'G:\咸海流域\24\BE\times_series\Processed_Band_1.csv',
    r'G:\咸海流域\24\BE\times_series\Processed_Band_2.csv',
    r'G:\咸海流域\24\BE\times_series\Processed_Band_3.csv',
    r'G:\咸海流域\24\BE\times_series\Processed_Band_4.csv',
]

# 读取标签索引（一次性，很小）
label_df = pd.read_csv(label_path, header=0, index_col=0)
selected_indices = set(label_df.index)  # 用 set 加速查找

# 每块读取行数，可调（越大越快，越占内存）
chunksize = 100000

# 处理每个波段文件
for band_path, output_path in zip(band_paths, output_paths):
    first_chunk = True
    with pd.read_csv(band_path, header=0, index_col=0, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, desc=f"处理 {band_path.split('\\')[-1]}"):
            # 只保留标签索引匹配的行
            filtered = chunk.loc[chunk.index.intersection(selected_indices)]
            if not filtered.empty:
                # 第一次写入带表头，后续追加
                filtered.to_csv(output_path, mode='a', header=first_chunk)
                first_chunk = False

print("✅ 所有文件处理完成")