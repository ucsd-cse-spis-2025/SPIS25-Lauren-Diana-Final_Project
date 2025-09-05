import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# Import data

# Tesla data
tesla_names = ["Date", "Open", "High", "Low", "Close/Last", "Volume"]
tesla_path = "data/tesla-stock-data.csv"

# Apple data
apple_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
apple_path = "data/apple-stock-data.csv"

# Nvidia data
nvidia_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
nvidia_path = "data/nvidia-stock-data.csv"

# Google data
google_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
google_path = "data/google-stock-data.csv"

# Meta data
meta_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
meta_path = "data/meta-stock-data.csv"

# Qualcomm data
qc_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
qc_path = "data/qualcomm-stock-data.csv"

# Microsoft data
ms_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
ms_path = "data/microsoft-stock-data.csv"

# Amazon data
amazon_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
amazon_path = "data/amazon-stock-data.csv"

# Samsung data
samsung_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
samsung_path = "data/samsung-stock-data.csv"

# Netflix data
netflix_names = ["Date", "Open", "High", "Low", "Close", "Volume"]
netflix_path = "data/netflix-stock-data.csv"


# Process the data and add columns
def data_cleaning(stock, path, col_names):
    stock_list = ["tesla", "apple", "nvidia", "google", "meta", "qualcomm", "microsoft", "amazon", "samsung", "netflix"]

    df = pd.read_csv(path)

    date = col_names[0]
    opens = col_names[1]
    high = col_names[2]
    low = col_names[3]
    close = col_names[4]
    volume = col_names[5]

    df["stock_id"] = stock_list.index(stock)

    # Tesla dataset where coloumn is called "Close/Last"
    if close == "Close/Last":
        df["Close"] = df["Close/Last"]
        close = "Close"

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

    # Feature engineering
    df['open-close'] = df[opens] - df[close]
    df['low-high'] = df[low] - df[high]
    df['daily_return'] = df[close].pct_change()
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

    df.dropna(inplace=True)

    # Normalize the data (just in case)
    for col in ['open-close', 'low-high', 'daily_return']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Create target
    df['target'] = np.where(df[close].shift(-1) > df[close], 1, 0)
    df.dropna(inplace=True)

    return df

# Declare features and scale
def data_processing(df, scaler=None, fit_scaler=True):
    feature_cols = ['open-close', 'low-high', 'daily_return', 'Volume', 'is_quarter_end', 'stock_id']
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

# Create train and test dataframes using combined data
paths = [tesla_path, apple_path, nvidia_path, google_path, meta_path, qc_path, ms_path, amazon_path, samsung_path, netflix_path]
names = [tesla_names, apple_names, nvidia_names, google_names, meta_names, qc_names, ms_names, amazon_names, samsung_names, netflix_names]
stock_list = ["tesla", "apple", "nvidia", "google", "meta", "qualcomm", "microsoft", "amazon", "samsung", "netflix"]

train_df = pd.DataFrame()
test_df = pd.DataFrame()

for i in range(len(paths)):
    df = data_cleaning(stock_list[i], paths[i], names[i])
    
    split_index = int(len(df) * 0.8)
    train_df = pd.concat([train_df, df.iloc[:split_index]], ignore_index=True)
    test_df = pd.concat([test_df, df.iloc[split_index:]], ignore_index=True)

# Scale and define features for training and validation sets
X_train, y_train, scaler = data_processing(train_df)
X_valid, y_valid, s = data_processing(test_df, scaler=scaler, fit_scaler=False)

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

# Print future prediction from user input
def predict_from_input(stock_name, model, scaler, open_price, high, low, close, volume, date_str):
    stock_list = ["tesla", "apple", "nvidia", "google", "meta", "qualcomm", "microsoft", "amazon", "samsung", "netflix"]

    # Process date
    date = pd.to_datetime(date_str)
    month = date.month

    # Features (must match training!)
    open_close = open_price - close
    low_high = low - high
    daily_return = 0  # can't compute with 1 data point
    is_quarter_end = 1 if month % 3 == 0 else 0
    stock_id = stock_list.index(stock_name)

    features = np.array([[open_close, low_high, daily_return, volume, is_quarter_end, stock_id]])
    features_scaled = scaler.transform(features)

    # Convert to tensor
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

        if prob > 0.5:
            prediction = "UP"
        else:
            prediction = "DOWN"
            prob = 1 - prob

        if prob == 1.0:
            prob = .99

    return prediction, prob

# Test the model with new data

# Process the data
df = data_cleaning("tesla", tesla_path, tesla_names)
X_test_scaled, y_test, s = data_processing(df, scaler=scaler, fit_scaler=False)

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



# Print future prediction from the current dataset
def predict_next_day(stock, model, scaler, path, col_names):
    # Only get latest row of data
    df = data_cleaning(stock, path, col_names)
    X_new_scaled, y_new, s = data_processing(df, scaler=scaler, fit_scaler=False)

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

