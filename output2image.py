import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


predict_path = r'G:\咸海流域\24\VH\pth\TCRNN1_output_result_VH_1.csv'
predict_df = pd.read_csv(predict_path, header=0, index_col=0)
print(predict_df)

# 将DataFrame转换为numpy数组
arr = predict_df.values

# 将数组转换为图像对象
img = Image.fromarray(np.uint8(arr), mode='L')

# 保存图像到本地
img.save(r'G:\咸海流域\24\VH\pth\TCRNN1_output_result_VH_1.tif')

