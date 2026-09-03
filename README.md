# GTN: Multi-Temporal Crop Classification with Gated Temporal Network

**六通道版本（4光谱 + 有效性掩码 + DOY） + Soft-DTW 一致性正则 + 门控注意力**

本仓库提供论文中使用的完整代码，支持模块化消融实验。

---

## 核心代码文件

| 文件名 | 类型 | 说明 |
|--------|------|------|
| **`MTCRNN_train_6ch_SoftDTW.py`** | **核心训练** | 完整六通道训练代码（含 Soft-DTW、门控注意力、T-LSTM、DOY） |
| **`MTCRNN_predict_6ch_SoftDTW.py`** | **核心预测** | 与训练严格对应的大规模预测代码 |
| `merge_times_series.py` | 数据融合 | 多区域时间序列对齐与合并（论文中用于扩大训练样本） |

---

## 数据预处理与后处理脚本

| 文件名 | 说明 |
|--------|------|
| `RGB_to_csv_batch_bands.py` | 多时相 TIFF → 按波段时间序列 CSV |
| `PolygonToRaster.py` | 矢量标签 → 栅格 |
| `label_process2.py` | 标签栅格 → 标签 CSV |
| `create_dataset_based_on_label_2.py` | 根据标签索引提取训练样本 |
| `output2image.py` | 预测结果 CSV → TIFF 影像 |
| `Consistent_projection.py` | 为输出影像赋予正确投影与地理变换 |

---

## 方法简介

- **输入**：六通道时间序列  
  1-4：光谱波段  
  5：时间有效性掩码（mask）  
  6：归一化 DOY（Day of Year）
- **网络结构**：1D CNN + T-LSTM + 门控注意力（Gated Attention）
- **正则**：Soft-DTW 一致性损失（时间增强双视角）
- **支持消融**：通过 `config` 开关控制 mask / time / gate / softdtw / doy

---

## 环境依赖

```bash
pip install torch numpy pandas scikit-learn tqdm
# GDAL 推荐使用 conda
conda install -c conda-forge gdal
