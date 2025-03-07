import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 🔹 Load the cleaned data
file_path = "C:/backend/archieve/cleaned_air_quality_data.csv"
df = pd.read_csv(file_path)

# 🔹 Ensure Date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by="Date")  # Sort by Date

# 🔹 Use AQI values for training
aqi_values = df["AQI"].values.reshape(-1, 1)

# 🔹 Normalize AQI values
scaler = MinMaxScaler(feature_range=(0, 1))
aqi_scaled = scaler.fit_transform(aqi_values)

# 🔹 Prepare training data (last 10 days to predict next day)
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(aqi_scaled)

# 🔹 Split into train & test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 🔹 Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# 🔹 Train Model
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))

# 🔹 Save Model
model_path = "C:/backend/fullstack/aqi_lstm_model.h5"
model.save(model_path)
print(f"✅ Model saved at {model_path}")
