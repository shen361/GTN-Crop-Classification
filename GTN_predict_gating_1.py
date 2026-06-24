import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import re
from datetime import datetime
import gc
import math
from torch.serialization import add_safe_globals
import warnings

# 忽略 Pandas 的性能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

'''TCRNN1预测代码 - 优化版，适用于大规模遥感影像时间序列数据'''
# --------------------------------------------------------------------- #
# 遥感影像的大小
height = 10980  # 行数，可能需要根据新时间序列调整
width = 10980  # 列数，可能需要根据新时间序列调整

# 分类类别数
num_class = 7  # 类别数

# 每次预测的数量（批处理大小）
batch_size = 5000

# 新增：指定数据分割数量
split_num = 7  # 将数据分割成 split_num 份处理，设置为 1 则保持原功能

# 加载完整时间序列数据路径
img_band1_path = r'G:\咸海流域\24\ND\times_series_left\Band_1.csv'
img_band2_path = r'G:\咸海流域\24\ND\times_series_left\Band_2.csv'
img_band3_path = r'G:\咸海流域\24\ND\times_series_left\Band_3.csv'
img_band4_path = r'G:\咸海流域\24\ND\times_series_left\Band_4.csv'

# 权重路径 - 加载训练好的TCRNN1模型
pth_path = r"G:\咸海流域\24\FN\pth_dtw_gating\tcrnn_best.pth"

# 输出路径
predict_path = r'G:\咸海流域\24\ND\pth_left\TCRNN1_predict_result_ND_left_1.csv'
outputs_path = r"G:\咸海流域\24\ND\pth_left\TCRNN1_output_result_ND_left_1.csv"

# --------------------------------------------------------------------- #
print("开始加载时间序列数据...")


# 快速获取CSV文件行数
def get_csv_row_count(file_path):
    """快速统计CSV文件行数（不包括表头）"""
    try:
        with open(file_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # 减去表头
        return row_count
    except Exception as e:
        print(f"统计 {file_path} 行数时出错: {e}")
        return None


# 加载单块DataFrame
def load_df_block(file_path, skiprows, nrows, desc=""):
    """加载指定行范围的DataFrame，替换-255为0"""
    try:
        with tqdm(total=1, desc=f"加载 {desc}", ncols=100) as pbar:
            df = pd.read_csv(file_path, header=0, index_col=0, skiprows=skiprows, nrows=nrows)
            df = df.replace(-255, 0).astype(np.float32)
            pbar.update(1)
        return df
    except Exception as e:
        print(f"加载 {file_path} 块时出错: {e}")
        return None


# 获取行数并划分索引
print("统计CSV文件行数...")
row_counts = [
    get_csv_row_count(img_band1_path),
    get_csv_row_count(img_band2_path),
    get_csv_row_count(img_band3_path),
    get_csv_row_count(img_band4_path)
]
if any(count is None for count in row_counts):
    raise ValueError("无法获取一个或多个CSV文件的行数")
if len(set(row_counts)) > 1:
    raise ValueError("波段文件的行数不一致")
num_samples = row_counts[0]
print(f"样本数: {num_samples}")

# 获取列名（从第一个文件读取第一行）
with open(img_band1_path, 'r') as f:
    columns = pd.read_csv(f, nrows=0).columns

# 划分索引范围
split_size = math.ceil(num_samples / split_num)
split_ranges = [(i, min(i + split_size, num_samples)) for i in range(1, num_samples + 1, split_size)]

# 检查样本数和时间步数
time_steps = len(columns)
print(f"时间步数: {time_steps}")

# 验证图像尺寸
if num_samples != height * width:
    print(f"警告: 样本数 {num_samples} 与预期图像尺寸 {height} x {width} = {height * width} 不匹配")


# --------------------------------------------------------------------- #
# 时间间隔提取函数
def extract_date_from_column(col_name):
    """从列名中提取YYMMDD格式的日期并转换为datetime对象"""
    match = re.match(r"(\d{6})", col_name)
    if match:
        date_str = match.group(1)
        try:
            date_obj = datetime.strptime(f"20{date_str}", "%Y%m%d")
            return date_obj
        except ValueError:
            return None
    return None


def calculate_time_intervals(columns):
    """计算每列对应日期与前一列日期之间的时间间隔（以天为单位）"""
    print("计算时间间隔...")
    dates = []
    for col in columns:
        date_obj = extract_date_from_column(col)
        dates.append(date_obj)

    intervals = [0]
    for i in range(1, len(dates)):
        if dates[i - 1] is None or dates[i] is None:
            intervals.append(0)
        else:
            delta_days = (dates[i] - dates[i - 1]).days
            intervals.append(max(0, delta_days))

    print(f"时间间隔计算完成，共{len(intervals)}个时间点")
    return np.array(intervals, dtype=np.float32)


# 改进的时间间隔下采样函数
def downsample_time_intervals(time_intervals, factor=2):
    """更精确地下采样时间间隔"""
    downsampled = []
    for i in range(0, len(time_intervals), factor):
        if i + factor <= len(time_intervals):
            window_sum = sum(time_intervals[i:i + factor])
            downsampled.append(window_sum)
        else:
            window_sum = sum(time_intervals[i:])
            downsampled.append(window_sum)
    return np.array(downsampled)


# 计算时间间隔
time_intervals = calculate_time_intervals(columns)

# --------------------------------------------------------------------- #
print("准备数据进行预测...")


# 将 DataFrame 分块转换为 NumPy 数组
def convert_df_to_numpy(df, batch_size=500000):
    """将 DataFrame 转换为 NumPy 数组，分块处理以节省内存"""
    num_samples = df.shape[0]
    time_steps = df.shape[1]
    arr = np.zeros((num_samples, time_steps), dtype=np.float32)

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_indices = slice(i, batch_end)
        batch_data = df.iloc[batch_indices].values.astype(np.float32)
        batch_data[batch_data == -255] = 0
        arr[batch_indices] = batch_data

        del batch_data
        gc.collect()

    return arr


# 计算时间衰减因子（向量化 + 分块）
def compute_time_decay_vectorized(band1, band2, band3, band4, batch_size=5000):
    """向量化计算时间衰减因子，分块处理以优化内存"""
    num_samples, time_steps = band1.shape
    time_decay = np.zeros((num_samples, time_steps), dtype=np.float32)

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_indices = slice(i, batch_end)

        mask1 = (band1[batch_indices] != -255)
        mask2 = (band2[batch_indices] != -255)
        mask3 = (band3[batch_indices] != -255)
        mask4 = (band4[batch_indices] != -255)

        valid_mask = (mask1 | mask2 | mask3 | mask4).astype(np.float32)
        time_decay[batch_indices] = valid_mask

        del mask1, mask2, mask3, mask4, valid_mask
        gc.collect()

    return time_decay


# 分批处理数据并合并
def process_data_in_batches(band1_array, band2_array, band3_array, band4_array, time_decay_factors, batch_size=5000):
    """分批处理数据，生成三维数组并合并通道"""
    num_samples, time_steps = band1_array.shape
    total_batches = math.ceil(num_samples / batch_size)

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_indices = slice(i, batch_end)

        with tqdm(total=5, desc=f"处理批次 {i // batch_size + 1}/{total_batches}", ncols=100) as pbar:
            band1_batch = band1_array[batch_indices].reshape(-1, time_steps, 1)
            pbar.update(1)
            band2_batch = band2_array[batch_indices].reshape(-1, time_steps, 1)
            pbar.update(1)
            band3_batch = band3_array[batch_indices].reshape(-1, time_steps, 1)
            pbar.update(1)
            band4_batch = band4_array[batch_indices].reshape(-1, time_steps, 1)
            pbar.update(1)
            time_decay_batch = time_decay_factors[batch_indices].reshape(-1, time_steps, 1)
            pbar.update(1)

        batch_dataset = np.concatenate(
            (band1_batch, band2_batch, band3_batch, band4_batch, time_decay_batch),
            axis=2
        )

        del band1_batch, band2_batch, band3_batch, band4_batch, time_decay_batch
        gc.collect()

        yield batch_dataset


# 定义T-LSTM单元
class TLSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TLSTM_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wi = nn.Linear(input_size, hidden_size, bias=True)
        self.Ui = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wf = nn.Linear(input_size, hidden_size, bias=True)
        self.Uf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wo = nn.Linear(input_size, hidden_size, bias=True)
        self.Uo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wc = nn.Linear(input_size, hidden_size, bias=True)
        self.Uc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_delta = nn.Linear(1, hidden_size, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev, delta_t):
        delta_t = delta_t.view(-1, 1)
        T = self.sigmoid(self.W_delta(delta_t))
        c_hat = c_prev * T

        i = self.sigmoid(self.Wi(x) + self.Ui(h_prev))
        f = self.sigmoid(self.Wf(x) + self.Uf(h_prev))
        o = self.sigmoid(self.Wo(x) + self.Uo(h_prev))
        c_tilde = self.tanh(self.Wc(x) + self.Uc(h_prev))

        c_new = f * c_hat + i * c_tilde
        h_new = o * self.tanh(c_new)

        return h_new, c_new


# 实现整个T-LSTM层
class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, dropout=0.2):
        super(TLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.cell = TLSTM_Cell(input_size, hidden_size)

    def forward(self, x, delta_t, h0=None, c0=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2)
            delta_t = delta_t.permute(1, 0)

        batch_size, seq_len, _ = x.size()

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        h, c = h0, c0

        for t in range(seq_len):
            x_t = x[:, t, :]
            delta_t_t = delta_t[:, t].unsqueeze(1)
            h, c = self.cell(x_t, h, c, delta_t_t)
            h = self.dropout(h)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c)


# 定义网络
class TCRNN1(nn.Module):
    def __init__(self, input_shape, time_steps, num_classes):
        super(TCRNN1, self).__init__()
        self.input_shape = input_shape
        self.time_steps = time_steps

        self.conv1 = nn.Conv1d(input_shape, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, 256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(32, 256)

        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self.tlstm1 = TLSTM(256, 256, batch_first=True, dropout=0.2)
        self.tlstm2 = TLSTM(256, 256, batch_first=True, dropout=0.2)

        self.attention = nn.Linear(256, 1)
        # 新增 gate
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, time_intervals):
        batch_size = x.size(0)
        x_conv = x.permute(0, 2, 1)

        x_conv = self.leaky_relu(self.gn1(self.conv1(x_conv)))
        x_conv = self.maxpool(x_conv)
        x_conv = self.leaky_relu(self.gn2(self.conv2(x_conv)))
        x_conv = self.leaky_relu(self.gn3(self.conv3(x_conv)))
        x_conv = self.leaky_relu(self.gn4(self.conv4(x_conv)))

        x_conv = x_conv.permute(0, 2, 1)
        time_intervals_downsampled = downsample_time_intervals(time_intervals, factor=2)

        if len(time_intervals_downsampled) < x_conv.size(1):
            pad_length = x_conv.size(1) - len(time_intervals_downsampled)
            time_intervals_downsampled = np.pad(time_intervals_downsampled, (0, pad_length), 'constant')
        elif len(time_intervals_downsampled) > x_conv.size(1):
            time_intervals_downsampled = time_intervals_downsampled[:x_conv.size(1)]

        delta_t = torch.tensor(time_intervals_downsampled, device=x_conv.device, dtype=torch.float32)
        delta_t = delta_t.unsqueeze(0).expand(batch_size, -1)

        x_lstm, _ = self.tlstm1(x_conv, delta_t)
        x_lstm, _ = self.tlstm2(x_lstm, delta_t)

        # attention gating
        gate_logits = self.gate(x_lstm)
        gates = torch.sigmoid(gate_logits)
        att_logits = self.attention(x_lstm)
        att_logits = att_logits * gates
        att_weights = F.softmax(att_logits, dim=1)
        context = torch.sum(x_lstm * att_weights, dim=1)

        x_fc = self.leaky_relu(self.fc1(context))
        x_fc = self.dropout(x_fc)
        x_fc = self.fc2(x_fc)

        return x_fc



# 预测函数
def predict_in_batches(model, data_generator, time_intervals, batch_size, device, num_samples, split_idx, split_total):
    """分批预测函数，接受生成器输入，添加分割部分的进度条"""
    model.eval()
    predictions = np.array([])

    total_batches = math.ceil(num_samples / batch_size)

    with tqdm(total=total_batches, desc=f"预测进度 (分割 {split_idx}/{split_total})", ncols=100) as pbar:
        for batch_data in data_generator:
            inputs = torch.from_numpy(batch_data).float().to(device)
            with torch.no_grad():
                outputs = model(inputs, time_intervals)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu().numpy()

            predictions = np.concatenate([predictions, predicted]) if len(predictions) > 0 else predicted

            torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)

    return predictions


# 主预测流程
def main():
    print("开始加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        import numpy as np
        from numpy._core.multiarray import _reconstruct
        from numpy import ndarray
        add_safe_globals([_reconstruct, ndarray])
        checkpoint = torch.load(pth_path, map_location=device)
    except Exception as e:
        print(f"使用安全全局变量加载失败: {e}")
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)

    saved_time_intervals = checkpoint.get('time_intervals', time_intervals)
    if saved_time_intervals is not None and len(saved_time_intervals) != len(time_intervals):
        print(f"警告: 当前时间序列长度({len(time_intervals)})与模型训练时的长度({len(saved_time_intervals)})不匹配")
        model_time_intervals = saved_time_intervals
    else:
        model_time_intervals = time_intervals

    input_shape = 5
    time_steps = len(columns)
    model = TCRNN1(input_shape=input_shape, time_steps=time_steps, num_classes=num_class).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载完成! 来自epoch {checkpoint.get('epoch', 'unknown')}")

    # 存储所有预测结果和索引
    all_predictions = np.array([])
    all_indices = []

    # 逐块加载和处理
    for i, (start_row, end_row) in enumerate(split_ranges, 1):
        print(f"处理分割 {i}/{len(split_ranges)}...")
        nrows = end_row - start_row

        # 加载当前块的DataFrame
        band1_df = load_df_block(img_band1_path, start_row, nrows, f"Band 1 分割 {i}")
        band2_df = load_df_block(img_band2_path, start_row, nrows, f"Band 2 分割 {i}")
        band3_df = load_df_block(img_band3_path, start_row, nrows, f"Band 3 分割 {i}")
        band4_df = load_df_block(img_band4_path, start_row, nrows, f"Band 4 分割 {i}")

        if any(df is None for df in [band1_df, band2_df, band3_df, band4_df]):
            raise ValueError(f"分割 {i} 无法加载一个或多个波段文件")

        split_index = band1_df.index

        # 转换为NumPy数组
        with tqdm(total=4, desc=f"转换分割 {i} 数据", ncols=100) as pbar:
            band1_array = convert_df_to_numpy(band1_df)
            pbar.update(1)
            band2_array = convert_df_to_numpy(band2_df)
            pbar.update(1)
            band3_array = convert_df_to_numpy(band3_df)
            pbar.update(1)
            band4_array = convert_df_to_numpy(band4_df)
            pbar.update(1)

        # 计算时间衰减因子
        print(f"计算时间衰减因子（分割 {i}/{len(split_ranges)}）...")
        with tqdm(total=1, desc=f"时间衰减因子 分割 {i}", ncols=100) as pbar:
            time_decay = compute_time_decay_vectorized(band1_array, band2_array, band3_array, band4_array,
                                                       batch_size=5000)
            pbar.update(1)

        # 生成数据并预测
        data_generator = process_data_in_batches(band1_array, band2_array, band3_array, band4_array, time_decay,
                                                 batch_size)
        split_predictions = predict_in_batches(
            model, data_generator, model_time_intervals, batch_size, device,
            band1_array.shape[0], i, len(split_ranges)
        )

        all_predictions = np.concatenate([all_predictions, split_predictions]) if len(
            all_predictions) > 0 else split_predictions
        all_indices.append(split_index)

        # 清理当前块的内存
        del band1_df, band2_df, band3_df, band4_df, band1_array, band2_array, band3_array, band4_array, time_decay, data_generator
        gc.collect()

    # 释放模型内存
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 合并索引
    index = np.concatenate(all_indices) if len(all_indices) > 1 else all_indices[0]

    print("保存预测结果...")
    dataframe = pd.DataFrame(data=all_predictions, index=index, columns=['value'])
    dataframe.to_csv(predict_path)
    print(f"预测结果已保存到: {predict_path}")

    print("将预测结果转换为图像格式...")
    img_dataframe = pd.DataFrame(index=range(height), columns=range(width))

    print("处理预测结果中...")
    with tqdm(total=len(dataframe.index), desc="转换为图像格式", ncols=100) as pbar:
        for i in dataframe.index:
            k = i
            i = i.replace('(', ' ').replace(')', ' ')
            values = i.split(',')
            try:
                img_dataframe.iloc[int(values[0]), int(values[1])] = int(dataframe.loc[k, "value"])
            except IndexError:
                print(f"警告: 索引 {i} 超出图像尺寸 {height}x{width}，跳过")
                continue
            pbar.update(1)

    print("保存图像格式结果...")
    img_dataframe.to_csv(outputs_path)
    print(f"图像格式结果已保存到: {outputs_path}")

    total_time = time.time() - start_time
    print(f"预测完成! 总用时: {total_time / 60:.2f}分钟")


if __name__ == "__main__":
    start_time = time.time()
    main()