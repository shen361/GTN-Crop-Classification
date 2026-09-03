import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re
from datetime import datetime
import math

# =========================================================
# 开关（0/1）—— 必须与预测时保持完全一致
# =========================================================
config = {
    "mask": 1,      # 是否使用时间有效性掩码
    "time": 1,      # 是否使用 T-LSTM（否则退化为普通 LSTM）
    "gate": 1,      # 是否使用门控注意力
    "softdtw": 1,   # 是否使用 Soft-DTW 一致性正则
    "doy": 1        # 是否加入 DOY 特征通道（六通道关键）
}

# Soft-DTW 相关超参
SOFTDTW_GAMMA = 0.1
SOFTDTW_WEIGHT = 0.1          # Soft-DTW 损失权重
AUG_SHIFT = 4                 # 时间偏移最大范围
AUG_STRETCH = (0.85, 1.15)    # 时间拉伸范围

# =========================================================
# 路径（请修改为你的实际路径）
# =========================================================
label_path = r'G:\dataset\Ablation\model\A\times_series\label_1.csv'
band1_path = r'G:\dataset\Ablation\model\A\times_series\Processed_Band_1.csv'
band2_path = r'G:\dataset\Ablation\model\A\times_series\Processed_Band_2.csv'
band3_path = r'G:\dataset\Ablation\model\A\times_series\Processed_Band_3.csv'
band4_path = r'G:\dataset\Ablation\model\A\times_series\Processed_Band_4.csv'

savepth_path = r"G:\dataset\Ablation\model\A\ab1\pth\MTCRNN_6ch_SoftDTW"
os.makedirs(savepth_path, exist_ok=True)

# =========================================================
# 指标计算
# =========================================================
def compute_metrics(y_true, y_pred, num_class):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_class)))
    # 如果有背景类（通常为0），可选择去掉
    if cm.shape[0] > 1:
        cm = cm[1:, 1:]
    oa = np.sum(np.diag(cm)) / (np.sum(cm) + 1e-8)
    ious, f1s = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        ious.append(tp / (tp + fp + fn + 1e-8))
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-8))
    return oa, np.mean(ious), np.mean(f1s)

# =========================================================
# 时间相关函数
# =========================================================
def extract_date(col):
    m = re.match(r"(\d{6})", col)
    if m:
        return datetime.strptime("20" + m.group(1), "%Y%m%d")
    return None

def calc_intervals(columns):
    dates = [extract_date(c) for c in columns]
    intervals = [0]
    for i in range(1, len(dates)):
        if dates[i] and dates[i-1]:
            intervals.append(max(0, (dates[i] - dates[i-1]).days))
        else:
            intervals.append(0)
    return np.array(intervals, dtype=np.float32)

def calc_doy(columns):
    doys = []
    for c in columns:
        d = extract_date(c)
        doys.append(d.timetuple().tm_yday / 365.0 if d else 0.0)
    return np.array(doys, dtype=np.float32)

# =========================================================
# Soft-DTW（稳定实现）
# =========================================================
def soft_dtw_distance(z1, z2, gamma=0.1):
    """z1, z2: (B, T, D) → 返回 (B,)"""
    B, T, D = z1.shape
    C = torch.cdist(z1, z2, p=2.0) ** 2
    R = torch.full((B, T + 1, T + 1), 1e9, device=z1.device, dtype=z1.dtype)
    R[:, 0, 0] = 0.0
    for i in range(1, T + 1):
        r0 = R[:, i-1, 1:T+1]
        r1 = R[:, i-1, 0:T]
        r2 = R[:, i, 0:T]
        c  = C[:, i-1, :]
        stack = torch.stack([r0, r1, r2], dim=-1)
        R[:, i, 1:T+1] = -gamma * torch.logsumexp(-(stack + c.unsqueeze(-1)) / gamma, dim=-1)
    return R[:, T, T]

# =========================================================
# 时间增强（仅对光谱通道操作）
# =========================================================
def time_augment(x, max_shift=4, stretch_range=(0.85, 1.15)):
    """x: (B, T, C)"""
    B, T, C = x.shape
    out = x.clone()

    # 1. 时间偏移
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    for b in range(B):
        s = shifts[b].item()
        if s > 0:
            out[b, s:, :4] = x[b, :T-s, :4]
            out[b, :s, :4] = x[b, 0:1, :4]
        elif s < 0:
            s = -s
            out[b, :T-s, :4] = x[b, s:, :4]
            out[b, T-s:, :4] = x[b, -1:, :4]

    # 2. 时间拉伸（简单线性插值近似）
    scales = torch.empty(B, device=x.device).uniform_(*stretch_range)
    for b in range(B):
        scale = scales[b].item()
        if abs(scale - 1.0) < 1e-3:
            continue
        new_T = max(2, int(T * scale))
        idx = torch.linspace(0, T-1, new_T, device=x.device)
        idx_floor = idx.long().clamp(0, T-1)
        idx_ceil  = (idx_floor + 1).clamp(0, T-1)
        w = (idx - idx_floor.float()).unsqueeze(1)
        stretched = (1 - w) * x[b, idx_floor, :4] + w * x[b, idx_ceil, :4]
        # 再插值回原长度
        idx2 = torch.linspace(0, new_T-1, T, device=x.device)
        idx2_floor = idx2.long().clamp(0, new_T-1)
        idx2_ceil  = (idx2_floor + 1).clamp(0, new_T-1)
        w2 = (idx2 - idx2_floor.float()).unsqueeze(1)
        out[b, :, :4] = (1 - w2) * stretched[idx2_floor] + w2 * stretched[idx2_ceil]

    return out

# =========================================================
# 数据加载
# =========================================================
print("加载数据...")
label_df = pd.read_csv(label_path, index_col=0)
band1_df = pd.read_csv(band1_path, index_col=0)
band2_df = pd.read_csv(band2_path, index_col=0)
band3_df = pd.read_csv(band3_path, index_col=0)
band4_df = pd.read_csv(band4_path, index_col=0)

label = label_df.iloc[:, 0].values
columns = band1_df.columns
time_intervals = calc_intervals(columns)

# mask
time_decay = np.zeros_like(band1_df.values, dtype=np.float32)
for i in range(band1_df.shape[1]):
    mask = (
        (band1_df.iloc[:, i] != -255) |
        (band2_df.iloc[:, i] != -255) |
        (band3_df.iloc[:, i] != -255) |
        (band4_df.iloc[:, i] != -255)
    )
    time_decay[:, i] = mask.astype(np.float32)

if not config["mask"]:
    time_decay = np.ones_like(time_decay)

# DOY
doy = None
if config["doy"]:
    doy = calc_doy(columns)
    doy = np.tile(doy[None, :], (band1_df.shape[0], 1))

# 填充缺失
for df in [band1_df, band2_df, band3_df, band4_df]:
    df[df == -255] = 0

channels = [
    band1_df.values[..., None],
    band2_df.values[..., None],
    band3_df.values[..., None],
    band4_df.values[..., None],
    time_decay[..., None]
]
if config["doy"]:
    channels.append(doy[..., None])

dataset = np.concatenate(channels, axis=2).astype(np.float32)
input_dim = dataset.shape[2]
print(f"输入通道数: {input_dim}  (mask={config['mask']}, doy={config['doy']})")

train_data, test_data, train_label, test_label = train_test_split(
    dataset, label, test_size=0.3, random_state=42)

num_class = int(np.max(label) + 1)
print(f"类别数: {num_class}")

# =========================================================
# 模型定义
# =========================================================
class TLSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wc = nn.Linear(input_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wd = nn.Linear(1, hidden_size)

    def forward(self, x, h, c, dt):
        dt = dt.view(-1, 1)
        T = torch.sigmoid(self.Wd(dt))
        c = c * T
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))
        o = torch.sigmoid(self.Wo(x) + self.Uo(h))
        g = torch.tanh(self.Wc(x) + self.Uc(h))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class TLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = TLSTM_Cell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, dt):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], h, c, dt[:, t])
            outputs.append(h)
        return torch.stack(outputs, 1)

class MTCRNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_shape, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv1d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.LeakyReLU(0.1)

        if config["time"]:
            self.rnn1 = TLSTM(256, 256)
            self.rnn2 = TLSTM(256, 256)
        else:
            self.rnn1 = nn.LSTM(256, 256, batch_first=True)
            self.rnn2 = nn.LSTM(256, 256, batch_first=True)

        self.attention = nn.Linear(256, 1)
        self.gate = nn.Linear(256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def _encode(self, x, dt):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.permute(0, 2, 1)

        # 时间间隔下采样对齐（与 pool 对应）
        dt_ds = dt[::2]
        if len(dt_ds) < x.size(1):
            dt_ds = F.pad(dt_ds, (0, x.size(1) - len(dt_ds)))
        elif len(dt_ds) > x.size(1):
            dt_ds = dt_ds[:x.size(1)]
        dt_batch = dt_ds.unsqueeze(0).expand(x.size(0), -1)

        if config["time"]:
            x = self.rnn1(x, dt_batch)
            x = self.rnn2(x, dt_batch)
        else:
            x, _ = self.rnn1(x)
            x, _ = self.rnn2(x)
        return x

    def forward(self, x, dt, return_feat=False):
        feat = self._encode(x, dt)
        att_raw = self.attention(feat)
        if config["gate"]:
            gate = torch.sigmoid(self.gate(feat))
            att = torch.softmax(att_raw * gate, dim=1)
        else:
            att = torch.softmax(att_raw, dim=1)
        context = torch.sum(feat * att, dim=1)
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        logits = self.fc2(x)
        if return_feat:
            return logits, feat
        return logits

# =========================================================
# 训练
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MTCRNN(input_dim, num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

dt_tensor = torch.tensor(time_intervals).float().to(device)

batch_size = 64
epochs = 30
best_oa = 0.0

print("开始训练...")
for epoch in range(epochs):
    model.train()
    preds, gts = [], []
    total_loss = 0.0

    loop = tqdm(range(0, len(train_data), batch_size),
                desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

    for i in loop:
        if i + batch_size > len(train_data):
            continue

        x = torch.from_numpy(train_data[i:i+batch_size]).float().to(device)
        y = torch.from_numpy(train_label[i:i+batch_size]).long().to(device)

        optimizer.zero_grad()

        if config["softdtw"]:
            # 双视角：原始 + 时间增强
            x_aug = time_augment(x, max_shift=AUG_SHIFT, stretch_range=AUG_STRETCH)
            out, feat = model(x, dt_tensor, return_feat=True)
            out_aug, feat_aug = model(x_aug, dt_tensor, return_feat=True)

            ce_loss = criterion(out, y)
            dtw_loss = soft_dtw_distance(feat, feat_aug, gamma=SOFTDTW_GAMMA).mean()
            loss = ce_loss + SOFTDTW_WEIGHT * dtw_loss
        else:
            out = model(x, dt_tensor)
            loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        preds.append(pred.cpu().numpy())
        gts.append(y.cpu().numpy())
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    oa, miou, f1 = compute_metrics(gts, preds, num_class)
    print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(loop):.4f} | OA: {oa:.4f} | mIoU: {miou:.4f} | Macro-F1: {f1:.4f}")

    if oa > best_oa:
        best_oa = oa
        torch.save({
            'model_state_dict': model.state_dict(),
            'time_intervals': time_intervals,
            'num_class': num_class,
            'input_dim': input_dim,
            'config': config
        }, os.path.join(savepth_path, "mtcrnn_6ch_softdtw_best.pth"))
        print(f"  → 保存最优模型 (OA={oa:.4f})")

print("训练完成！")
