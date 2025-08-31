import kagglehub

path = kagglehub.dataset_download("iamtanmayshukla/tesla-stocks-dataset")
print("Path to dataset files:", path)


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import ta


df = pd.read_csv(r"C:\Users\laure\.cache\kagglehub\datasets\iamtanmayshukla\tesla-stocks-dataset\versions\5\HistoricalData_1726367135218.csv")
#df = df[df["Name"] == "AAPL"]


names = ['Close/Last', 'Open', 'High', 'Low']

for col in names:
  df[col] = df[col].replace('[/$]', '', regex=True).astype(float)


splitted = df['Date'].str.split('/', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')


# Feature engineering
df['open-close'] = df['Open'] - df['Close/Last']
df['low-high'] = df['Low'] - df['High']
df['daily_return'] = df['Close/Last'].pct_change()
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)


df['rsi'] = ta.momentum.RSIIndicator(df['Close/Last'], window=14).rsi()
df['macd'] = ta.trend.MACD(df['Close/Last']).macd_diff()
df['bollinger_h'] = ta.volatility.BollingerBands(df['Close/Last']).bollinger_hband()
df['bollinger_l'] = ta.volatility.BollingerBands(df['Close/Last']).bollinger_lband()
df['ema_12'] = ta.trend.EMAIndicator(df['Close/Last'], window=12).ema_indicator()
df['ema_26'] = ta.trend.EMAIndicator(df['Close/Last'], window=26).ema_indicator()


# Drop NA from pct_change
df = df.dropna()


df.head()


# Target: 1 if next day close is higher
df['target'] = np.where(df['Close/Last'].shift(-1) > df['Close/Last'], 1, 0)
df = df.dropna()

# Define feature columns
feature_cols = ['open-close', 'low-high', 'daily_return', 'Volume', 'is_quarter_end']
X = df[feature_cols]
y = df['target']


# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train/Test Split
train_size = int(len(X_scaled) * 0.8)
X_train, X_valid = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_valid = y[:train_size], y[train_size:]


# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)


class StrongerMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = StrongerMLP(input_dim=X_train.shape[1])


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        train_preds = torch.sigmoid(outputs)
        train_preds_cls = (train_preds > 0.5).float()
        train_acc = accuracy_score(y_train_tensor.numpy(), train_preds_cls.numpy())

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid_tensor)
        val_preds = torch.sigmoid(val_outputs)
        val_preds_cls = (val_preds > 0.5).float()
        val_acc = accuracy_score(y_valid_tensor, val_preds_cls)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")


model.eval()
with torch.no_grad():
    preds = model(X_valid_tensor)
    preds_prob = torch.sigmoid(preds)
    preds_cls = (preds_prob > 0.5).float()

print("Classification Report:")
print(classification_report(y_valid_tensor, preds_cls))

