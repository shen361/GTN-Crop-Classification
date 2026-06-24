import numpy as np
from osgeo import gdal, gdalconst
import os
import pandas as pd


def read_img(img_path):
    """读取遥感数据信息"""
    dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    img_width = dataset.RasterXSize  # 读取栅格数据集的X方向像素数
    img_height = dataset.RasterYSize  # 读取栅格数据集的Y方向像素数
    img_bands = dataset.RasterCount  # 读取栅格数据集的波段数

    band_1 = dataset.GetRasterBand(1)  # 读取波段1

    adf_GeoTransform = dataset.GetGeoTransform()  # 仿射矩阵
    im_Proj = dataset.GetProjection()  # 地图投影信息
    # 将数据写成数组
    img_data = np.array(dataset.ReadAsArray())  # 三维数组
    data = np.array(band_1.ReadAsArray())  # 二维数组
    print(data.size)
    del dataset
    return img_data, data, img_width, img_height, img_bands, adf_GeoTransform, im_Proj


if __name__ == "__main__":
    img_root = r'G:\咸海流域\24\BE\shp_to_tif\12.tif'
    csv_root = r'G:\咸海流域\24\BE\times_series\label_1.csv'
    delete_values = [66]  # 修改为需要删除的值列表

    # 创建DataFrame
    img_data, data, img_width, img_height, img_bands, adf_GeoTransform, im_Proj = read_img(img_root)
    print("img_width :{} img_height:{} img_bands:{}".format(img_width, img_height, img_bands))

    data1 = pd.DataFrame(np.array(data))
    a = []
    b = []

    for i in range(img_height):
        for j in range(img_width):
            a.append(int(data1.iloc[i, j]))
            b.append((i, j))

    df_merge = pd.DataFrame(data=a, index=b, columns=['label'])

    # 删除满足指定值的行
    print('-----正在删除 value 值为 {} 的所有行-----'.format(delete_values))
    data_filtered = df_merge[~df_merge.label.isin(delete_values)]

    # 定义类别数
    num_class = np.max(data_filtered['label']) + 1  # 类别数从0开始计数
    print('类别数：{}'.format(num_class))

    # 将标签导出
    data_filtered.to_csv(csv_root)
    print('csv保存完毕！')
