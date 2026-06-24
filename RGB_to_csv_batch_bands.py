import numpy as np
from osgeo import gdal, gdalconst
import os
import pandas as pd
from tqdm import tqdm


def read_img(img_path, band_num):
    """高效读取遥感数据指定波段"""
    dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    if not dataset:
        raise ValueError(f"无法打开文件：{img_path}")
    band = dataset.GetRasterBand(band_num)
    data = band.ReadAsArray()
    img_width, img_height = dataset.RasterXSize, dataset.RasterYSize
    dataset = None  # 显式释放资源
    return data, img_width, img_height


if __name__ == "__main__":
    img_root = r'G:\咸海流域\24\VH\original_tif'          # 原始影像目录
    output_dir = r'G:\咸海流域\24\VH\times_series'         # 输出 CSV 目录（自动按波段建子目录）
    os.makedirs(output_dir, exist_ok=True)

    name_list = [f for f in os.listdir(img_root) if f.lower().endswith('.tif')]
    if not name_list:
        raise RuntimeError("未找到任何 tif 文件，请检查 img_root 路径")
    print(f"发现 {len(name_list)} 个 TIFF 文件")

    # 用第一张影像获取波段总数
    sample_path = os.path.join(img_root, name_list[0])
    ds = gdal.Open(sample_path, gdalconst.GA_ReadOnly)
    band_count = ds.RasterCount
    ds = None

    # 遍历每个波段
    for b in range(1, band_count + 1):
        print(f"\n正在处理第 {b} 波段 ……")

        # 初始化索引结构（以第一张影像为准）
        first_data, base_width, base_height = read_img(sample_path, b)
        index = pd.MultiIndex.from_product(
            [range(base_height), range(base_width)],
            names=['row', 'col']
        )
        df = pd.DataFrame(index=index)
        df.index = df.index.map(lambda x: f"({x[0]}, {x[1]})")
        df.index.name = 'coordinates'

        # 批量处理文件
        for filename in tqdm(name_list, desc=f"构建波段{b}时间序列"):
            file_path = os.path.join(img_root, filename)
            data, width, height = read_img(file_path, b)

            if (width != base_width) or (height != base_height):
                print(f"尺寸不符已跳过: {filename}")
                continue

            df[filename[:-4]] = data.ravel()

        # 输出结果
        csv_name = f"Band_{b}.csv"
        output_path = os.path.join(output_dir, csv_name)
        df.to_csv(output_path)
        print(f"波段 {b} 的 CSV 文件已生成: {output_path}")