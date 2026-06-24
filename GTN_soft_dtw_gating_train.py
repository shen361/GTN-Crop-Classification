import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import re
import math
import tsaug

# ===============================
# 训练脚本（加入时间增强 + Soft-DTW 一致性正则 + Attention Gate）
# ===============================

# --------------------------------------------------------------------- #
# 配置路径（与原脚本一致）
label_path = r'G:\咸海流域\24\BE\times_series\label_1.csv'
band1_path = r'G:\咸海流域\24\BE\times_series\Processed_Band_1.csv'
band2_path = r'G:\咸海流域\24\BE\times_series\Processed_Band_2.csv'
band3_path = r'G:\咸海流域\24\BE\times_series\Processed_Band_3.csv'
band4_path = r'G:\咸海流域\24\BE\times_series\Processed_Band_4.csv'

# 模型保存路径
savepth_path = r"G:\咸海流域\24\BE\pth_dtw_gating"
os.makedirs(savepth_path, exist_ok=True)

# --------------------------------------------------------------------- #
# 加载数据
print("正在加载数据...")
label_df = pd.read_csv(label_path, header=0, index_col=0)
band1_df = pd.read_csv(band1_path, header=0, index_col=0)
band2_df = pd.read_csv(band2_path, header=0, index_col=0)
band3_df = pd.read_csv(band3_path, header=0, index_col=0)
band4_df = pd.read_csv(band4_path, header=0, index_col=0)

# --------------------------------------------------------------------- #
# 获取时间信息和生成时间差
def extract_date_from_column(col_name):
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
    return np.array(intervals, dtype=np.float32)

def downsample_time_intervals(time_intervals, factor=2):
    downsampled = []
    for i in range(0, len(time_intervals), factor):
        if i + factor <= len(time_intervals):
            window_sum = sum(time_intervals[i:i + factor])
            downsampled.append(window_sum)
        else:
            window_sum = sum(time_intervals[i:])
            downsampled.append(window_sum)
    return np.array(downsampled)

print("计算时间间隔...")
columns = band1_df.columns
time_intervals = calculate_time_intervals(columns)
print(f"时间间隔计算完成，共{len(time_intervals)}个时间点")

# --------------------------------------------------------------------- #
# 准备训练数据
print("准备训练数据...")
label = label_df.iloc[:, 0].values

train_data_band1 = band1_df.values
train_data_band2 = band2_df.values
train_data_band3 = band3_df.values
train_data_band4 = band4_df.values

# 缺失值处理（-255 -> 0）
train_data_band1[train_data_band1 == -255] = 0
train_data_band2[train_data_band2 == -255] = 0
train_data_band3[train_data_band3 == -255] = 0
train_data_band4[train_data_band4 == -255] = 0

# 时间衰减因子
time_decay_factors = np.zeros_like(train_data_band1)
for i in range(train_data_band1.shape[1]):
    mask1 = (band1_df.iloc[:, i].values != -255)
    mask2 = (band2_df.iloc[:, i].values != -255)
    mask3 = (band3_df.iloc[:, i].values != -255)
    mask4 = (band4_df.iloc[:, i].values != -255)
    valid_mask = mask1 | mask2 | mask3 | mask4
    time_decay_factors[:, i] = valid_mask.astype(np.float32)

# 重塑为三维
train_data_band1 = train_data_band1.reshape((train_data_band1.shape[0], train_data_band1.shape[1], 1))
train_data_band2 = train_data_band2.reshape((train_data_band2.shape[0], train_data_band2.shape[1], 1))
train_data_band3 = train_data_band3.reshape((train_data_band3.shape[0], train_data_band3.shape[1], 1))
train_data_band4 = train_data_band4.reshape((train_data_band4.shape[0], train_data_band4.shape[1], 1))
time_decay_factors = time_decay_factors.reshape((time_decay_factors.shape[0], time_decay_factors.shape[1], 1))

dataset = np.concatenate((train_data_band1, train_data_band2, train_data_band3, train_data_band4, time_decay_factors),
                         axis=2)

train_data, test_data, train_label, test_label = train_test_split(dataset, label, test_size=0.3, random_state=42)
print(f'训练数据形状：{train_data.shape}')
print(f'测试数据形状：{test_data.shape}')
num_class = np.max(label) + 1
print(f'类别数：{num_class}')

# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# T-LSTM 单元（与原脚本一致）
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
        Tt = self.sigmoid(self.W_delta(delta_t))
        c_hat = c_prev * Tt
        i = self.sigmoid(self.Wi(x) + self.Ui(h_prev))
        f = self.sigmoid(self.Wf(x) + self.Uf(h_prev))
        o = self.sigmoid(self.Wo(x) + self.Uo(h_prev))
        c_tilde = self.tanh(self.Wc(x) + self.Uc(h_prev))
        c_new = f * c_hat + i * c_tilde
        h_new = o * self.tanh(c_new)
        return h_new, c_new

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

# --------------------------------------------------------------------- #
# 定义网络（增加 Attention Gate）
class TCRNN(nn.Module):
    def __init__(self, input_shape, time_steps, num_classes):
        super(TCRNN, self).__init__()
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
        self.gate = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_common(self, x, time_intervals):
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
        return x_lstm

    def forward(self, x, time_intervals):
        x_lstm = self._forward_common(x, time_intervals)
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

    def forward_with_features(self, x, time_intervals):
        x_lstm = self._forward_common(x, time_intervals)
        gate_logits = self.gate(x_lstm)
        gates = torch.sigmoid(gate_logits)
        att_logits = self.attention(x_lstm)
        att_logits = att_logits * gates
        att_weights = F.softmax(att_logits, dim=1)
        context = torch.sum(x_lstm * att_weights, dim=1)
        x_fc = self.leaky_relu(self.fc1(context))
        x_fc = self.dropout(self.fc2(x_fc))
        return x_fc, x_lstm, gates


# --------------------------------------------------------------------- #
# Soft-DTW（批量版）：基于 Cuturi & Blondel 公式的稳定实现（O(B*T*T)）
def soft_dtw_distance(z1, z2, gamma=0.1):
    """
    z1, z2: (B, T, D) 的序列特征
    返回：每个样本的 soft-DTW 标量 (B,)
    """
    assert z1.shape[:2] == z2.shape[:2], "两视角序列长度必须一致"
    B, T, D = z1.shape
    C = torch.cdist(z1, z2, p=2.0) ** 2  # (B, T, T)

    inf = 1e9
    R = torch.full((B, T + 1, T + 1), inf, device=z1.device, dtype=z1.dtype)
    R[:, 0, 0] = 0.0

    for i in range(1, T + 1):
        r0 = R[:, i - 1, 1: T + 1]
        r1 = R[:, i - 1, 0: T]
        r2 = R[:, i,     0: T]
        c  = C[:, i - 1, :]
        stack = torch.stack([r0, r1, r2], dim=-1)  # (B,T,3)
        val = -gamma * torch.logsumexp(-(stack + c.unsqueeze(-1)) / gamma, dim=-1)
        R[:, i, 1:T + 1] = val

    return R[:, T, T]


# --------------------------------------------------------------------- #
# 时间增强（二视角）：shift / stretch / drop（仅对前4个光谱通道做，不动时间衰减通道）
def _time_shift(x, max_shift=6):
    if max_shift <= 0:
        return x
    B, T, C = x.shape
    shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(B,), device=x.device)
    out = x.clone()
    for b in range(B):
        s = shifts[b].item()
        if s > 0:
            out[b, s:, :4] = x[b, :T - s, :4]
            out[b, :s, :4] = x[b, 0:1, :4]
        elif s < 0:
            s = -s
            out[b, :T - s, :4] = x[b, s:, :4]
            out[b, T - s:, :4] = x[b, -1:, :4]
    return out

def _time_stretch(x, scale_low=0.8, scale_high=1.2):
    B, T, C = x.shape
    scales = torch.empty(B, device=x.device).uniform_(scale_low, scale_high)
    base_grid = torch.linspace(0, 1, T, device=x.device).view(1, T)
    out = x.clone()
    for b in range(B):
        s = scales[b].item()
        src_pos = torch.clamp(base_grid * s, 0, 1)
        src_idx = src_pos * (T - 1)
        idx0 = torch.floor(src_idx).long()
        idx1 = torch.clamp(idx0 + 1, max=T - 1)
        w = (src_idx - idx0.float()).unsqueeze(-1)
        v0 = x[b, idx0, :4]
        v1 = x[b, idx1, :4]
        out[b, :, :4] = v0 * (1 - w) + v1 * w
    return out

def _time_drop(x, drop_prob=0.15):
    if drop_prob <= 0:
        return x
    B, T, C = x.shape
    out = x.clone()
    for b in range(B):
        mask = (torch.rand(T, device=x.device) < drop_prob)
        if mask.any():
            idx = torch.where(mask)[0]
            for t in idx.tolist():
                if t == 0:
                    out[b, t, :4] = x[b, t + 1, :4]
                elif t == T - 1:
                    out[b, t, :4] = x[b, t - 1, :4]
                else:
                    out[b, t, :4] = 0.5 * (x[b, t - 1, :4] + x[b, t + 1, :4])
    return out

def build_time_aug_2views(x,
                          p_shift=0.7, p_stretch=0.7, p_drop=0.7,
                          max_shift=6, scale_low=0.85, scale_high=1.15, drop_prob=0.15):
    x2 = x.clone()
    if torch.rand(1).item() < p_shift:
        x2 = _time_shift(x2, max_shift=max_shift)
    if torch.rand(1).item() < p_stretch:
        x2 = _time_stretch(x2, scale_low=scale_low, scale_high=scale_high)
    if torch.rand(1).item() < p_drop:
        x2 = _time_drop(x2, drop_prob=drop_prob)
    return x, x2

# --------------------------------------------------------------------- #
# 训练超参数
lr = 0.0003
batch_size = 64
epochs = 100
weight_decay = 1e-5
lambda_con = 0.2
softdtw_gamma = 0.1
lambda_gate = 1e-4   # <<< 新增 gate 正则权重

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型与优化器
input_shape = 5
time_steps = train_data.shape[1]
model = TCRNN(input_shape=input_shape, time_steps=time_steps, num_classes=num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

steps_per_epoch = max(1, len(train_data) // batch_size)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                          max_lr=lr,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          pct_start=0.3)

# 早停
early_stopping_patience = 10
early_stopping_counter = 0
best_val_loss = float('inf')

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print(f"开始训练T-CRNN模型(含 Soft-DTW 一致性 + Gate)，共{epochs}轮...")
start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for i in range(0, len(train_data), batch_size):
        if i + batch_size <= len(train_data):
            batch_data = train_data[i:i + batch_size].copy()
            batch_labels = train_label[i:i + batch_size]

            # tsaug 增强
            if np.random.random() < 0.7:
                bands_data = batch_data[:, :, :4]
                augment_choice = np.random.choice([1, 2, 3])
                if augment_choice == 1:
                    augmented_bands = tsaug.AddNoise(scale=0.02).augment(bands_data)
                elif augment_choice == 2:
                    augmented_bands = tsaug.Drift(max_drift=0.05, n_drift_points=5).augment(bands_data)
                else:
                    augmented_bands = tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=1.5).augment(bands_data)
                batch_data[:, :, :4] = augmented_bands

            inputs = torch.from_numpy(batch_data).float().to(device)
            labels = torch.from_numpy(batch_labels).long().to(device)

            # 两视角增强
            view1, view2 = build_time_aug_2views(inputs)

            optimizer.zero_grad()

            logits1, feat1, gates1 = model.forward_with_features(view1, time_intervals)
            logits2, feat2, gates2 = model.forward_with_features(view2, time_intervals)

            loss_cls = criterion(logits1, labels)
            B, Tprime, D = feat1.shape
            sdtw_vals = soft_dtw_distance(feat1, feat2, gamma=softdtw_gamma)
            loss_con = (sdtw_vals / max(1, Tprime)).mean()
            gate_reg = (gates1.mean() + gates2.mean()) * 0.5

            loss = loss_cls + lambda_con * loss_con + lambda_gate * gate_reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.shape[0]
            _, predicted = torch.max(logits1.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / max(1, train_total)
    train_acc = train_correct / max(1, train_total)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            if i + batch_size <= len(test_data):
                batch_data = test_data[i:i + batch_size]
                batch_labels = test_label[i:i + batch_size]

                inputs = torch.from_numpy(batch_data).float().to(device)
                labels = torch.from_numpy(batch_labels).long().to(device)

                outputs = model(inputs, time_intervals)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.shape[0]
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / max(1, val_total)
    val_acc = val_correct / max(1, val_total)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time_intervals': time_intervals,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss
        }, os.path.join(savepth_path, 'tcrnn_best.pth'))
        print(f"保存最佳模型，验证精度: {val_acc:.4f}, 验证损失: {val_loss:.4f}")
    else:
        early_stopping_counter += 1
        print(f"验证损失未改善，早停计数: {early_stopping_counter}/{early_stopping_patience}")

    if early_stopping_counter >= early_stopping_patience:
        print(f"早停触发于 epoch {epoch + 1}")
        break

    epoch_time = time.time() - epoch_start
    print(f'第 {epoch + 1}/{epochs} 轮 | 训练损失: {train_loss:.4f} | 训练精度: {train_acc:.4f} | '
          f'验证损失: {val_loss:.4f} | 验证精度: {val_acc:.4f} | 用时: {epoch_time:.1f}s')

# 保存最终模型
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'time_intervals': time_intervals,
    'final_val_acc': history['val_acc'][-1] if len(history['val_acc'])>0 else None
}, os.path.join(savepth_path, 'tcrnn_final.pth'))

total_time = time.time() - start_time
print(f'训练完成! 总用时: {total_time / 60:.2f}分钟')

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='训练损失')
plt.plot(history['val_loss'], label='验证损失')
plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend(); plt.title('损失曲线')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='训练精度')
plt.plot(history['val_acc'], label='验证精度')
plt.xlabel('轮次'); plt.ylabel('精度'); plt.legend(); plt.title('精度曲线')
plt.tight_layout()
plt.savefig(os.path.join(savepth_path, 'training_history.png'), dpi=300)
plt.close()
