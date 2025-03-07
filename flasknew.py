from flask import Flask, jsonify, request
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained LSTM model
model = tf.keras.models.load_model("aqi_lstm_model.h5")

# API Token & Base URL for Live AQI Data
AQI_API_TOKEN = "ac31025b9f13a21e86af011ad15c87832cd1d46e"
BASE_URL = "https://api.waqi.info/feed"

# Function to fetch live AQI data
def get_live_aqi(city):
    response = requests.get(f"{BASE_URL}/{city}/?token={AQI_API_TOKEN}")

    data = response.json()
    if data["status"] == "ok":
        return data["data"]["aqi"]
    return None

# Function to predict future AQI
def predict_future_aqi(past_aqi_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_aqi_values = np.array(past_aqi_values).reshape(-1, 1)
    past_aqi_scaled = scaler.fit_transform(past_aqi_values)

    X_test = np.array([past_aqi_scaled[-10:]])  # Using last 10 values
    predicted_aqi_scaled = model.predict(X_test)
    predicted_aqi = scaler.inverse_transform(predicted_aqi_scaled)
    
    return predicted_aqi.flatten().tolist()

@app.route("/get-aqi", methods=["GET"])
def get_aqi():
    city = request.args.get("city", "Bangalore")
    live_aqi = get_live_aqi(city)

    if live_aqi is None:
        return jsonify({"error": "Could not fetch AQI data"}), 500

    # Generate dummy past AQI values (Ideally, fetch from DB)
    past_aqi_values = [live_aqi - np.random.randint(1, 10) for _ in range(10)]
    future_aqi = predict_future_aqi(past_aqi_values)

    return jsonify({
        "city": city,
        "live_aqi": live_aqi,
        "future_aqi": future_aqi
    })

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, jsonify, request
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained LSTM model
model = tf.keras.models.load_model("C:/backend/fullstack/aqi_lstm_model.h5")  # Ensure correct path

# API Token & Base URL for Live AQI Data
AQI_API_TOKEN = "ac31025b9f13a21e86af011ad15c87832cd1d46e"
BASE_URL = "https://api.waqi.info/feed"

# Function to fetch live AQI data
def get_live_aqi(city):
    response = requests.get(f"{BASE_URL}/{city}/?token={AQI_API_TOKEN}")  # ✅ Correct API token usage
    data = response.json()
    if data["status"] == "ok":
        return data["data"]["aqi"]
    return None

# Function to predict future AQI
def predict_future_aqi(past_aqi_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    past_aqi_values = np.array(past_aqi_values).reshape(-1, 1)
    past_aqi_scaled = scaler.fit_transform(past_aqi_values)  # ✅ Scale properly

    X_test = np.array([past_aqi_scaled[-10:]])  # Using last 10 values
    predicted_aqi_scaled = model.predict(X_test)
    predicted_aqi = scaler.inverse_transform(predicted_aqi_scaled)  # ✅ Reverse scaling
    
    return predicted_aqi.flatten().tolist()

@app.route("/get-aqi", methods=["GET"])
def get_aqi():
    city = request.args.get("city", "Bangalore")
    live_aqi = get_live_aqi(city)

    if live_aqi is None:
        return jsonify({"error": "Could not fetch AQI data"}), 500

    # Generate dummy past AQI values (Ideally, fetch from DB)
    past_aqi_values = [live_aqi - np.random.randint(1, 10) for _ in range(10)]
    future_aqi = predict_future_aqi(past_aqi_values)

    return jsonify({
        "city": city,
        "live_aqi": live_aqi,
        "future_aqi": future_aqi
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # ✅ Add port 5000 for consistency
