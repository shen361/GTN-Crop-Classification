# MTCRNN: Multi-Temporal Crop Classification with Gated Temporal Network

**六通道版本（4光谱 + 有效性掩码 + DOY） + Soft-DTW 一致性正则 + 门控注意力**

本仓库提供论文中使用的完整代码，支持消融实验（通过开关控制各模块）。

## 主要特性

- **六通道输入**：4个光谱波段 + 时间有效性掩码 + 归一化DOY（Day of Year）
- **T-LSTM**：显式建模时间间隔（Δt）
- **门控注意力（Gated Attention）**
- **Soft-DTW 一致性正则**：对时间增强后的双视角特征施加 Soft-DTW 约束
- **模块化开关**：方便进行消融实验
- **多区域时间序列对齐融合工具**

## 环境依赖

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm matplotlib
# GDAL 推荐用 conda 安装
conda install -c conda-forge gdal
```
## 目录结构
```
├── MTCRNN_train_6ch_SoftDTW.py     # 完整训练代码（六通道 + Soft-DTW）
├── MTCRNN_predict_6ch_SoftDTW.py   # 对应预测代码
├── merge_times_series.py           # 多区域时间序列对齐与融合
├── RGB_to_csv_batch_bands.py       # TIFF → 时间序列 CSV
├── label_process2.py               # 标签栅格 → CSV
├── create_dataset_based_on_label_2.py
├── PolygonToRaster.py
├── output2image.py
├── Consistent_projection.py
└── README.md
```
## 快速开始
1. 数据准备流程

使用 RGB_to_csv_batch_bands.py 将多时相 TIFF 转为按波段的时间序列 CSV。
使用 PolygonToRaster.py + label_process2.py 生成标签 CSV。
使用 create_dataset_based_on_label_2.py 提取有标签的训练样本。
（可选）使用 merge_times_series.py 将多个区域的时间序列对齐并合并（论文中用于扩大训练样本）。

2. 训练
修改 MTCRNN_train_6ch_SoftDTW.py 顶部路径和 config 开关后运行：python MTCRNN_train_6ch_SoftDTW.py

config 开关说明：
开关,含义,推荐值
mask,是否使用时间有效性掩码,1
time,是否使用 T-LSTM,1
gate,是否使用门控注意力,1
softdtw,是否使用 Soft-DTW 正则,1
doy,是否加入 DOY 通道（六通道）,1

训练会自动保存最优模型 mtcrnn_6ch_softdtw_best.pth。
3. 预测
修改 MTCRNN_predict_6ch_SoftDTW.py 中路径，确保 config 与训练时完全一致，然后运行：python MTCRNN_predict_6ch_SoftDTW.py

4. 结果转影像
使用 output2image.py 将预测 CSV 转为 TIFF，再用 Consistent_projection.py 赋予正确投影信息。
多区域时间序列融合
merge_times_series.py 用于将多个不同区域、时间不完全一致的时间序列对齐到统一时间轴后合并，是论文中扩大训练样本的重要工具。
使用方法：修改脚本底部的 region_folders 列表和 output_folder 后直接运行即可。

引用
如果本代码对你的研究有帮助，请引用我们的论文：


注意事项

所有脚本中的路径均为示例路径，使用前请全部修改。
训练与预测时的 config 必须保持完全一致，否则会报错或结果错误。
列名需包含6位日期信息（YYMMDD），以便正确提取时间间隔和 DOY。
缺失值统一使用 -255 表示。
