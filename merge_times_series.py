import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# ===========================
# 自定义时间戳解析函数（统一6位编码）
# ===========================
def extract_date_key(col):
    match = re.match(r"(\d{6})", col)
    return match.group(1) if match else None

# ===========================
# 主函数：合成多个区域的四波段时间序列和标签
# ===========================
def merge_regions(region_folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    band_names = ['Processed_Band_1.csv', 'Processed_Band_2.csv', 'Processed_Band_3.csv', 'Processed_Band_4.csv']
    all_date_keys = set()
    date_key_to_fullcols = {}

    # 1. 收集所有区域中出现过的标准化时间键（6位数字）
    print("🔍 收集所有区域时间戳...")
    for region in region_folders:
        for band_file in band_names:
            df = pd.read_csv(os.path.join(region, band_file), index_col=0)
            for col in df.columns:
                key = extract_date_key(col)
                if key:
                    all_date_keys.add(key)
                    date_key_to_fullcols.setdefault(key, set()).add(col)

    all_date_keys = sorted(all_date_keys)
    print(f"✅ 标准时间点数: {len(all_date_keys)}")

    # 2. 初始化总表
    merged_bands = {band: pd.DataFrame() for band in band_names}
    merged_labels = pd.Series(dtype=int)

    # 3. 遍历所有区域进行对齐合成
    for region in tqdm(region_folders, desc="📦 合并区域"):
        region_bands = {}

        for band_file in band_names:
            df = pd.read_csv(os.path.join(region, band_file), index_col=0)
            aligned_df = pd.DataFrame(index=df.index, columns=all_date_keys)

            for key in all_date_keys:
                candidate_cols = list(date_key_to_fullcols.get(key, []))
                found = False
                for c in candidate_cols:
                    if c in df.columns:
                        aligned_df[key] = df[c]
                        found = True
                        break
                if not found:
                    aligned_df[key] = -255

            region_bands[band_file] = aligned_df

        # 添加到总表
        for band_file in band_names:
            region_df = region_bands[band_file]
            region_df.index = [f"{region}_{idx}" for idx in region_df.index]
            merged_bands[band_file] = pd.concat([merged_bands[band_file], region_df])

        # 标签处理
        label_path = os.path.join(region, 'label_1.csv')
        if os.path.exists(label_path):
            label_df = pd.read_csv(label_path, index_col=0)
            label_df = label_df.iloc[:, 0] if label_df.shape[1] == 1 else label_df.squeeze()
            label_df.index = [f"{region}_{idx}" for idx in label_df.index]
            merged_labels = pd.concat([merged_labels, label_df])

    # 4. 保存合并结果
    print("💾 保存合并后的时间序列和标签...")
    for band_file in band_names:
        merged_bands[band_file].to_csv(os.path.join(output_folder, band_file))
    merged_labels.to_csv(os.path.join(output_folder, 'label_1.csv'))

    print(f"✅ 合成完成，已保存至：{output_folder}")


# 示例用法：
if __name__ == "__main__":
    region_folders = [
        r"G:\咸海流域\24\BE\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        r"G:\咸海流域\24\BF\times_series",
        # r"G:\dataset\B_part_xianhai\multi\1\times_series",
        # r"G:\dataset\B_part_xianhai\multi\2\times_series",
        # r"G:\dataset\B_part_xianhai\multi\3\times_series",
        # r"E:\SangZhuZiQu\left\top_times_series",
        # r"E:\JiangZiXian\shang\original_image\times_series",
        # r"F:\DingRiXian\left\times_series_top",
        # r"F:\RiKaZe\GangBaXian\times_series_bot",
        # r"E:\SaJiaXian\dongbei\times_series",
        #  r"G:\dataset\A_part_rikaze\multi\1\times_series\renamed",
        # r"F:\RiKaZe\KangMaXian\times_series_top",
        # r"F:\RiKaZe\RenBuXian\left\times_series",
        # r"F:\RiKaZe\RenBuXian\right\times_series",
        # r"F:\RiKaZe\KangMaXian\times_series_bot",
        # r"F:\RiKaZe\BaiLangXian\xibei\times_series",
        # r"F:\RiKaZe\BaiLangXian\dongnan\times_series",
        # r"F:\RiKaZe\BaiLangXian\dongbei\times_series",
        # r"E:\SaJiaXian\xinan\times_series",
        # r"E:\SaJiaXian\dongnan\l_times_series",
        # r"E:\SaJiaXian\dongnan\r_times_series",
        #  r"F:\new_model\merge_times_series\1",
        # r"F:\new_model\merge_times_series\1",
        # r"G:\dataset\Ablation\model\B\ab1\times_series",
        # r"G:\dataset\Ablation\model\B\ab2\times_series",
        # r"G:\dataset\Ablation\model\B\ab3\times_series",
        # r"G:\dataset\A_part_rikaze\part1\bottom_right\times_series\renamed",
        # r"F:\changji\23\VJ\times_series",
        # r"F:\changji\23\VK\times_series",
        # r"F:\changji\23\WJ\times_series",
        # r"F:\changji\23\WK\times_series",
        # r"F:\changji\23\XJ\times_series",
        # r"F:\changji\23\XK\times_series",
        # r"F:\changji\23\YJ\times_series",
        # r"F:\changji\23\YK\times_series",
        # 添加更多路径...
    ]
    output_folder = r"F:\new_model\merge_times_series\5_for_exp\merge_for_tongji"
    merge_regions(region_folders, output_folder)
