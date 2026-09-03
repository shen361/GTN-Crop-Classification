import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
from datetime import datetime
from tqdm import tqdm

# ============================
# 开关（必须与训练完全一致！）
# ============================
config = {
    "mask": 1,
    "time": 1,
    "gate": 1,
    "softdtw": 1,   # 预测时不影响结构，但保持一致
    "doy": 1
}

# ============================
# 路径
# ============================
img_band1_path = r'你的路径\Band_1.csv'
img_band2_path = r'你的路径\Band_2.csv'
img_band3_path = r'你的路径\Band_3.csv'
img_band4_path = r'你的路径\Band_4.csv'

pth_path = r'你的路径\mtcrnn_6ch_softdtw_best.pth'
predict_path = r'你的路径\mtcrnn_predict_result.csv'

# ============================
# 时间函数（同训练）
# ============================
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

# ============================
# 模型（同训练）
# ============================
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

    def forward(self, x, dt):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.permute(0, 2, 1)

        dt_ds = dt[::2]
        if len(dt_ds) < x.size(1):
            pad = x.size(1) - len(dt_ds)
            dt_ds = torch.nn.functional.pad(dt_ds, (0, pad))
        elif len(dt_ds) > x.size(1):
            dt_ds = dt_ds[:x.size(1)]
        dt_batch = dt_ds.unsqueeze(0).expand(x.size(0), -1)

        if config["time"]:
            x = self.rnn1(x, dt_batch)
            x = self.rnn2(x, dt_batch)
        else:
            x, _ = self.rnn1(x)
            x, _ = self.rnn2(x)

        att_raw = self.attention(x)
        if config["gate"]:
            gate = torch.sigmoid(self.gate(x))
            att = torch.softmax(att_raw * gate, dim=1)
        else:
            att = torch.softmax(att_raw, dim=1)

        context = torch.sum(x * att, dim=1)
        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        return self.fc2(x)

# ============================
# 数据准备
# ============================
print("加载预测数据...")
band1_df = pd.read_csv(img_band1_path, index_col=0)
band2_df = pd.read_csv(img_band2_path, index_col=0)
band3_df = pd.read_csv(img_band3_path, index_col=0)
band4_df = pd.read_csv(img_band4_path, index_col=0)

columns = band1_df.columns
time_intervals = calc_intervals(columns)

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

doy = None
if config["doy"]:
    doy = calc_doy(columns)
    doy = np.tile(doy[None, :], (band1_df.shape[0], 1))

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

data = np.concatenate(channels, axis=2).astype(np.float32)
input_dim = data.shape[2]
print(f"输入通道数: {input_dim}")

# ============================
# 加载模型并预测
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(pth_path, map_location=device, weights_only=False)

num_class = checkpoint['num_class']
input_dim = checkpoint.get('input_dim', input_dim)

model = MTCRNN(input_dim, num_class).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dt = torch.tensor(checkpoint['time_intervals']).float().to(device)

preds = []
batch_size = 5000
for i in tqdm(range(0, data.shape[0], batch_size), desc="预测中"):
    batch = torch.from_numpy(data[i:i+batch_size]).float().to(device)
    with torch.no_grad():
        out = model(batch, dt)
        _, pred = torch.max(out, 1)
    preds.append(pred.cpu().numpy())

preds = np.concatenate(preds)
pd.DataFrame(preds, columns=['value']).to_csv(predict_path, index=False)
print("预测完成！结果已保存至:", predict_path)
