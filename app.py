import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import urllib.request
import os
import pandas as pd


st.set_page_config(page_title="Prediksi Suhu LSTM", layout="wide")


st.markdown("""
    <style>
    body, .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #38bdf8;
        font-weight: bold;
        text-align: center;
    }
    .stHeader {
        color: #38bdf8;
    }
    .css-1d391kg {
        background-color: #1e293b !important;
        color: white;
    }
    div.stButton > button {
        background-color: #38bdf8;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0ea5e9;
        color: white;
    }
    .stNumberInput input {
        background-color: #1e293b;
        color: white;
        border-radius: 5px;
    }
    .stDataFrame {
        background-color: #1e293b;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
            st.success(f"✅ Berhasil mengunduh {filename}")
        except Exception as e:
            st.error(f"❌ Gagal unduh {filename}: {e}")
            return False
    return True


SCALER_URL = "https://github.com/riosp-lab/riosp-lab/raw/main/scaler.pkl"
MODEL_URL = "https://github.com/riosp-lab/riosp-lab/raw/main/model.tflite"
DATA_URL = "https://github.com/riosp-lab/riosp-lab/raw/main/jena_climate_mini.csv"


st.title("🌡️ Prediksi Suhu Menggunakan LSTM")


with st.sidebar:
    st.header("🔧 Pengaturan")
    st.info("Model LSTM (.tflite) + Scaler digunakan untuk memprediksi suhu.")
    show_dataset = st.checkbox("📊 Tampilkan Dataset Contoh")


scaler_loaded = False
if download_file(SCALER_URL, "scaler.pkl"):
    try:
        scaler = joblib.load('scaler.pkl')
        scaler_loaded = True
        st.sidebar.success("✅ Scaler berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"❌ Scaler gagal dimuat: {e}")
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        dummy_data = np.array([[0], [1]])
        scaler.fit(dummy_data)
        st.sidebar.warning("⚠️ Menggunakan scaler dummy")


model_loaded = False
if download_file(MODEL_URL, "model.tflite"):
    try:
        interpreter = tf.lite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_loaded = True
        st.sidebar.success("✅ Model berhasil dimuat")
    except Exception as e:
        st.sidebar.error(f"❌ Model gagal dimuat: {e}")
        st.stop()


def predict(data):
    if not scaler_loaded or not model_loaded:
        st.error("❗ Model atau scaler belum tersedia.")
        return None

    try:
        data_scaled = scaler.transform(np.array(data).reshape(-1, 1))
        data_scaled = data_scaled.reshape(1, 10, 1).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], data_scaled)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0][0]
    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")
        return None


st.header("📝 Masukkan Data Suhu (10 Nilai Terakhir)")
cols = st.columns(5)
input_data = []

for i in range(10):
    with cols[i % 5]:
        value = st.number_input(
            f"Data ke-{i+1} (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=0.5,
            step=0.1,
            key=f"input_{i}"
        )
        input_data.append(value)


if st.button("🚀 Prediksi Sekarang", type="primary"):
    with st.spinner("Memproses..."):
        result = predict(input_data)

    if result is not None:
        st.success(f"🎯 Prediksi Suhu: **{result:.2f} °C**")
        st.balloons()


if show_dataset and download_file(DATA_URL, "jena_climate_mini.csv"):
    try:
        df = pd.read_csv("jena_climate_mini.csv")
        st.header("📁 Dataset Contoh")
        st.dataframe(df.head(), use_container_width=True)
        if st.checkbox("📈 Tampilkan Statistik"):
            st.write(df.describe())
    except Exception as e:
        st.error(f"❌ Gagal load dataset: {e}")
