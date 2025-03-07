import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load AQI Dataset
df = pd.read_csv("C:/backend/archieve/cleaned_air_quality_data.csv")
  # Use full path
# Ensure this dataset has past AQI values
aqi_values = df["AQI"].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
aqi_scaled = scaler.fit_transform(aqi_values.reshape(-1, 1))

# Create sequences for training
X, y = [], []
for i in range(10, len(aqi_scaled)):
    X.append(aqi_scaled[i-10:i])
    y.append(aqi_scaled[i])

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=20, batch_size=16)

# Save Model
model.save("aqi_lstm_model.h5")
