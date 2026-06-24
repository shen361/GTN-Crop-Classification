from osgeo import gdal, ogr, gdalconst
import datetime
from tqdm import tqdm
import os


def pol2line(polyfn, linefn):
    """
        This function is used to make polygon convert to line
    :param polyfn: the path of input, the shapefile of polygon
    :param linefn: the path of output, the shapefile of line
    :return:
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    polyds = ogr.Open(polyfn, 0)
    polyLayer = polyds.GetLayer()
    spatialref = polyLayer.GetSpatialRef()
    # 创建输出文件
    if os.path.exists(linefn):
        driver.DeleteDataSource(linefn)
    lineds = driver.CreateDataSource(linefn)
    linelayer = lineds.CreateLayer(linefn, srs=spatialref, geom_type=ogr.wkbLineString)
    featuredefn = linelayer.GetLayerDefn()
    # 获取ring到几何体
    # geomline = ogr.Geometry(ogr.wkbGeometryCollection)
    for feat in polyLayer:
        geom = feat.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        # geomcoll.AddGeometry(ring)
        outfeature = ogr.Feature(featuredefn)
        outfeature.SetGeometry(ring)
        linelayer.CreateFeature(outfeature)
        outfeature = None


def Rasterize(input_shp, input_tif, output_tif, field, filed_type, NoValue=-9999):
    """
    input_shp:需要转为栅格的矢量文件（矢量文件路径）
    input_tif:模板栅格，用于读取地理变换信息、栅格大小，将其应用于新的栅格上
    output_tif:输出栅格文件（栅格文件路径）
    field:字符串，栅格值的字段
    filed_type:栅格值类型，一般选择gdal.GDT_Int16,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64等几种类型
    NoValue:整型或浮点型，矢量空白区转换后的值
    """
    data = gdal.Open(input_tif, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    proj = data.GetProjection()
    h = data.RasterXSize
    w = data.RasterYSize

    driver = ogr.GetDriverByName("ESRI Shapefile")
    open_shp = driver.Open(input_shp, 1)
    shp_ly = open_shp.GetLayer()
    x_min, x_max, y_min, y_max = shp_ly.GetExtent()

    pixel_size = geo_transform[1]
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    target_ds = gdal.GetDriverByName('GTiff').Create(output_tif, w, h, 1, filed_type)

    target_ds.SetGeoTransform((x_min, pixel_size, 0.0, y_max, 0.0, -pixel_size))
    target_ds.SetProjection(proj)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()

    if field is None:
        gdal.RasterizeLayer(target_ds, [1], shp_ly, None)
        y_buffer = band.ReadAsArray()
    else:
        option = 'ATTRIBUTE=' + field
        gdal.RasterizeLayer(target_ds, [1], shp_ly, options=[option])

    target_ds = None


def shp_to_tiff(shp_file, output_tiff, attribute):
    """

    :param shp_file:
    :param output_tiff:
    :param attribute: 定义栅格值的矢量属性
    :return:
    """
    start_time = datetime.datetime.now()
    print("start :" + str(start_time))
    # 读取shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shp_file, 1)
    # 获取图层文件对象
    shp_layer = data_source.GetLayer()
    lon_min, lon_max, lat_min, lat_max = shp_layer.GetExtent()
    s_projection = str(shp_layer.GetSpatialRef())

    # (0,0,:,0,:,0)表示旋转系数
    # 自定义仿射矩阵系数 ， 10表示分辨率大小，决定了栅格像元的大小
    dst_transform = (lon_min, 10, 0, lat_max, 0, -10)
    # dst_transform = data_source.GetTransform()
    d_lon = int(abs((lon_max - lon_min) / dst_transform[1]))  # 除以横向分辨率
    d_lat = int(abs((lat_max - lat_min) / dst_transform[5]))

    # 根据模板tif属性信息创建对应标准的目标栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(output_tiff, d_lon, d_lat, 1, gdal.GDT_Int32)
    print(target_ds)
    target_ds.SetGeoTransform(dst_transform)
    target_ds.SetProjection(s_projection)

    band = target_ds.GetRasterBand(1)

    # 设置背景数值
    # NoData_value = 0
    # band.SetNoDataValue(NoData_value)
    # band.FlushCache()

    # 调用栅格化函数。gdal.RasterizeLayer函数有四个参数，分别有栅格对象，波段，矢量对象，value的属性值将为栅格值
    option = 'ATTRIBUTE=' + attribute
    gdal.RasterizeLayer(target_ds, [1], shp_layer, options=[option])
    # gdal.RasterizeLayer(target_ds, [1], shp_layer, None)
    # 直接写入
    y_buffer = band.ReadAsArray()
    print(y_buffer)
    target_ds.WriteRaster(0, 0, d_lon, d_lat, y_buffer.tobytes())
    start_time = datetime.datetime.now()
    print("end :" + str(start_time))
    target_ds = None  # todo 释放内存，只有强制为None才可以释放干净
    del target_ds, shp_layer


if __name__ == '__main__':
    # 输入完整名称，包括输出
    shp_path = r'G:\咸海流域\24\BE\shp_of_original_tif\240527.shp'
    save_path = r'G:\咸海流域\24\BE\shp_to_tif\12.tif'
    # temp_line_path = os.path.join(shp_path, 'temp')
    # name_list = [i for i in os.listdir(shp_path) if i.endswith('.shp')]
    # if not os.path.exists(temp_line_path):
    #     os.mkdir(temp_line_path)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)F:\RiKaZe\NanMuLinXian\xibei\shp_tif\230608xb.shp

    shp_to_tiff(shp_path, save_path, 'value')  # Max_gridco
    # Rasterize(shp_path, save_path, save_path, 'value', gdal.GDT_Byte)

# for name in tqdm(name_list[0:1]):
# pol2line(os.path.join(shp_path, name), os.path.join(temp_line_path, name))
# shp_to_tiff(os.path.join(temp_line_path, name), os.path.join(save_path, name.split('.')[0] + '.tif'), 1)
# shp_to_tiff(os.path.join(shp_path, name), os.path.join(save_path, name.split('.')[0] + '.tif'), 'FID_1')
# Rasterize(os.path.join(shp_path, name), os.path.join(shp_path, name.split('.')[0] + '.tif'), os.path.join(save_path, name.split('.')[0] + '.tif'), 'FID', gdal.GDT_Byte)
