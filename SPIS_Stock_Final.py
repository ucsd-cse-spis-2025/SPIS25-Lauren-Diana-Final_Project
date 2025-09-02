import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import ta
import matplotlib.pyplot as plt

import kagglehub

# Import data

# Tesla data
path1 = kagglehub.dataset_download("iamtanmayshukla/tesla-stocks-dataset")

print("Path to dataset files:", path1)

tesla_names = ["Date", "Open", "High", "Low", "Close/Last", "Volume"]
tesla_path = r"C:\Users\laure\.cache\kagglehub\datasets\iamtanmayshukla\tesla-stocks-dataset\versions\5\HistoricalData_1726367135218.csv"

# Apple data
path2 = kagglehub.dataset_download("varpit94/apple-stock-data-updated-till-22jun2021")

print("Path to dataset files:", path2)

apple_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
apple_path = r"C:\Users\laure\.cache\kagglehub\datasets\varpit94\apple-stock-data-updated-till-22jun2021\versions\8\AAPL.csv"

# Nvidia data
path3 = kagglehub.dataset_download("programmerrdai/nvidia-stock-historical-data")

print("Path to dataset files:", path3)

nvidia_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
nvidia_path = r"C:\Users\laure\.cache\kagglehub\datasets\programmerrdai\nvidia-stock-historical-data\versions\1\NVDA (1).csv"

# Google data
path4 = kagglehub.dataset_download("henryshan/google-stock-price")

print("Path to dataset files:", path4)

google_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
google_path = r"C:\Users\laure\.cache\kagglehub\datasets\henryshan\google-stock-price\versions\1\GOOG.csv"

# Meta data
path5 = kagglehub.dataset_download("umerhaddii/meta-stock-data-2025")

print("Path to dataset files:", path5)

meta_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
meta_path = r"C:\Users\laure\.cache\kagglehub\datasets\umerhaddii\meta-stock-data-2025\versions\2\META stocks.csv"

# Qualcomm data
path6 = kagglehub.dataset_download("varunsaikanuri/qualcomm-stocks-historical-data")

print("Path to dataset files:", path6)

qc_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
qc_path = r"C:\Users\laure\.cache\kagglehub\datasets\varunsaikanuri\qualcomm-stocks-historical-data\versions\19\Qualcomm_Stocks.csv"

# Microsoft data
path7 = kagglehub.dataset_download("umerhaddii/microsoft-stock-data-2025")

print("Path to dataset files:", path7)

ms_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
ms_path = r"C:\Users\laure\.cache\kagglehub\datasets\umerhaddii\microsoft-stock-data-2025\versions\1\MSFT_1986-03-13_2025-02-04.csv"

# Amazon data
path8 = kagglehub.dataset_download("adilshamim8/amazon-stock-price-history")

print("Path to dataset files:", path8)

amazon_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
amazon_path = r"C:\Users\laure\.cache\kagglehub\datasets\adilshamim8\amazon-stock-price-history\versions\7\Amazon_stock_data.csv"

# Samsung data
path9 = kagglehub.dataset_download("caesarmario/samsung-electronics-stock-historical-price")

print("Path to dataset files:", path9)

samsung_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
samsung_path = r"C:\Users\laure\.cache\kagglehub\datasets\caesarmario\samsung-electronics-stock-historical-price\versions\873\005930.KS.csv"

# Netflix data
path0 = kagglehub.dataset_download("adilshamim8/netflix-stock-price-history")

print("Path to dataset files:", path0)

netflix_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
netflix_path = r"C:\Users\laure\.cache\kagglehub\datasets\adilshamim8\netflix-stock-price-history\versions\8\Netflix_stock_data.csv"

# Process the data, add features, and scale
def data_process(path, col_names, scaler=None, fit_scaler=True):
    df = pd.read_csv(path)

    date = col_names[0]
    opens = col_names[1]
    high = col_names[2]
    low = col_names[3]
    close = col_names[4]
    volume = col_names[5]

    if "Close/Last" in col_names:
        df["Close"] = df["Close/Last"]

    # Clean columns
    names = [close, opens, high, low]
    for col in names:
        df[col] = df[col].replace('[/$]', '', regex=True).astype(float)

    # Date features
    if '-' in df[date][0]:
        splitted = df[date].str.split('-', expand=True)
        df['day'] = splitted[2].astype(int)
        df['month'] = splitted[1].astype(int)
        df['year'] = splitted[0].astype(int)
    elif '/' in df[date][0]:
        splitted = df[date].str.split('/', expand=True)
        df['day'] = splitted[1].astype(int)
        df['month'] = splitted[0].astype(int)
        df['year'] = splitted[2].astype(int)

    df[date] = pd.to_datetime(df[date])

    # Feature engineering
    df['open-close'] = df[opens] - df[close]
    df['low-high'] = df[low] - df[high]
    df['daily_return'] = df[close].pct_change()
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

    df.dropna(inplace=True)

    # Create target
    df['target'] = np.where(df[close].shift(-1) > df[close], 1, 0)
    df.dropna(inplace=True)

    feature_cols = ['open-close', 'low-high', 'daily_return', volume, 'is_quarter_end']
    X = df[feature_cols]
    y = df['target']

    # Scale features
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler

# Call the processing function on all stocks
X_scaled, y, scaler = data_process(tesla_path, tesla_names)

# Combine all stocks into one dataframe


# Train/Test Split
train_size = int(len(X_scaled) * 0.8)
X_train, X_valid = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_valid = y[:train_size], y[train_size:]


# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)


class MLP(nn.Module):
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

model = MLP(input_dim=X_train.shape[1])

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

'''
# Test the model with new data

# Process the data
X_test_scaled, y_test, s, df = data_process(nvidia_path, nvidia_names, scaler=scaler, fit_scaler=False)

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Set the model to evaluate
model.eval()

# Get predictions
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_probs = torch.sigmoid(test_outputs)
    test_preds_cls = (test_probs > 0.5).float()

# Print classification report
from sklearn.metrics import classification_report, accuracy_score

print("Test Accuracy:", accuracy_score(y_test_tensor, test_preds_cls))
print("Classification Report on Test Data:")
print(classification_report(y_test_tensor, test_preds_cls))
'''

# Print future prediction from the current dataset

def predict_next_day(model, scaler, path, col_names):
    # Only get latest row of data
    X_new_scaled, y_new, s = data_process(path, col_names, scaler=scaler, fit_scaler=False)

    # Use the yesterday's features
    latest_features = X_new_scaled[-1].reshape(1, -1)
    latest_tensor = torch.tensor(latest_features, dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(latest_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "UP" if prob > 0.5 else "DOWN"

    return prediction, prob

# Print future prediction from user input

def predict_from_input(model, scaler, open_price, high, low, close, volume, date_str):

    # Process date
    date = pd.to_datetime(date_str)
    month = date.month

    # Features (must match training!)
    open_close = open_price - close
    low_high = low - high
    daily_return = 0  # can't compute with 1 data point
    is_quarter_end = 1 if month % 3 == 0 else 0

    features = np.array([[open_close, low_high, daily_return, volume, is_quarter_end]])
    features_scaled = scaler.transform(features)

    # Convert to tensor
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "UP" if prob > 0.5 else "DOWN"

    return prediction, prob