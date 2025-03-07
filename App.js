import React, { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const App = () => {
  const [aqiData, setAqiData] = useState(null);
  const [predictedData, setPredictedData] = useState([]);
  // eslint-disable-next-line no-unused-vars
  const [location, setLocation] = useState({ lat: 12.9716, lng: 77.5946 }); // Default: Bangalore

  useEffect(() => {
    fetchLiveAQI();
    fetchPredictedAQI();
  }, []);

  const fetchLiveAQI = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/live-aqi");
      setAqiData(response.data);
    } catch (error) {
      console.error("Error fetching live AQI:", error);
    }
  };

  const fetchPredictedAQI = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/predict-aqi");
      setPredictedData(response.data.predictions);
    } catch (error) {
      console.error("Error fetching predicted AQI:", error);
    }
  };

  return (
    <div>
      <h1>🌍 AQI Improvement System</h1>
      <h2>Live AQI: {aqiData ? aqiData.aqi : "Loading..."}</h2>
      
      {/* OpenStreetMap */}
      <MapContainer center={[location.lat, location.lng]} zoom={12} style={{ height: "400px", width: "100%" }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        <Marker position={[location.lat, location.lng]}>
          <Popup>Live AQI: {aqiData ? aqiData.aqi : "Loading..."}</Popup>
        </Marker>
      </MapContainer>

      {/* AQI Prediction Graph */}
      <h2>📈 AQI Prediction for Next Few Days</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={predictedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="aqi" stroke="#8884d8" activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default App;