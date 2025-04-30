import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import urllib.request
import os

# Konfigurasi awal
st.set_page_config(page_title="Prediksi Suhu LSTM", layout="wide")

# Fungsi untuk download file dari GitHub
@st.cache_resource
def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
            st.success(f"Berhasil mengunduh {filename}!")
        except Exception as e:
            st.error(f"Gagal mengunduh {filename}: {e}")
            return False
    return True

# URL file (Ganti dengan URL raw GitHub Anda)
SCALER_URL = "https://github.com/username/repo/raw/main/scaler.pkl"
MODEL_URL = "https://github.com/username/repo/raw/main/model.tflite"
DATA_URL = "https://github.com/username/repo/raw/main/jena_climate_mini.csv"

# Judul Aplikasi
st.title("🌡️ Prediksi Suhu dengan LSTM (TFLite)")

# Sidebar untuk info tambahan
with st.sidebar:
    st.header("Pengaturan")
    st.info("Aplikasi ini memprediksi suhu menggunakan model LSTM yang dikonversi ke TFLite")
    show_dataset = st.checkbox("Tampilkan dataset contoh")

# 1. Load Scaler
scaler_loaded = False
if download_file(SCALER_URL, "scaler.pkl"):
    try:
        scaler = joblib.load('scaler.pkl')
        scaler_loaded = True
        st.sidebar.success("Scaler berhasil dimuat!")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat scaler: {e}")
        # Fallback scaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        dummy_data = np.array([[0], [1]])
        scaler.fit(dummy_data)
        st.sidebar.warning("Menggunakan scaler dummy!")

# 2. Load Model
model_loaded = False
if download_file(MODEL_URL, "model.tflite"):
    try:
        interpreter = tf.lite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_loaded = True
        st.sidebar.success("Model berhasil dimuat!")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")
        st.stop()

# Fungsi Prediksi
def predict(data):
    if not scaler_loaded or not model_loaded:
        st.error("Model/scaler belum siap!")
        return None
    
    try:
        # Preprocessing
        data_scaled = scaler.transform(np.array(data).reshape(-1, 1))
        data_scaled = data_scaled.reshape(1, 10, 1).astype(np.float32)
        
        # Inference
        interpreter.set_tensor(input_details[0]['index'], data_scaled)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0][0]
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        return None

# Input Data
st.header("Input Data Prediksi")
cols = st.columns(5)
input_data = []

for i in range(10):
    with cols[i%5]:
        value = st.number_input(
            f"Data ke-{i+1} (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=0.5,
            step=0.1,
            key=f"input_{i}"
        )
        input_data.append(value)

# Tombol Prediksi
if st.button("🚀 Prediksi Sekarang", type="primary"):
    with st.spinner("Memproses..."):
        prediction = predict(input_data)
        
    if prediction is not None:
        st.success(f"**Hasil Prediksi:** {prediction:.2f} °C")
        st.balloons()

# Tampilkan Dataset (Opsional)
if show_dataset and download_file(DATA_URL, "jena_climate_mini.csv"):
    try:
        import pandas as pd
        df = pd.read_csv('jena_climate_mini.csv')
        st.header("Contoh Dataset")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.checkbox("Tampilkan Statistik"):
            st.write(df.describe())
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")