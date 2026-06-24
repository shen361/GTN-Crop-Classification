import os
from osgeo import gdal

# 打开要设置投影坐标的图像F:\RiKaZe\GangBaXian\pth\bot\CRNN_output_1.tif
input_path = r'G:\咸海流域\24\VH\pth\TCRNN1_output_result_VH_1.tif'
input_ds = gdal.Open(input_path)

# 打开用于参考的图像
reference_path = r"G:\咸海流域\24\VH\original_tif\240602.tif"
reference_ds = gdal.Open(reference_path)

# 获取参考图像的投影和仿射变换
reference_proj = reference_ds.GetProjection()
reference_transform = reference_ds.GetGeoTransform()

# 创建输出图像
output_path = r'G:\咸海流域\24\VH\pth\P_TCRNN1_output_result_VH_1.tif'
output_driver = gdal.GetDriverByName("GTiff")
output_ds = output_driver.CreateCopy(output_path, input_ds)

# 设置输出图像的投影和仿射变换
output_ds.SetProjection(reference_proj)
output_ds.SetGeoTransform(reference_transform)

# 关闭数据集
input_ds = None
reference_ds = None
output_ds = None

# 删除输入数据集，以免在下次运行时出现错误
# os.remove(input_path + ".aux.xml")
