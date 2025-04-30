import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Muat scaler
try:
    scaler = joblib.load('scaler.pkl')
    st.success("Scaler berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat scaler: {e}")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dummy_data = np.array([[0], [1]])
    scaler.fit(dummy_data)
    joblib.dump(scaler, 'scaler.pkl')
    st.warning("Scaler baru telah dibuat dan disimpan!")

# Muat model TFLite
try:
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("Model TFLite berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# Fungsi untuk preprocessing dan prediksi
def predict(data):
    data_scaled = scaler.transform(np.array(data).reshape(-1, 1))
    data_scaled = data_scaled.reshape(1, 10, 1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], data_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

# Antarmuka Streamlit
st.title("Prediksi Suhu dengan LSTM (TFLite)")
st.write("Masukkan 10 data terakhir untuk memprediksi suhu berikutnya:")

# Input data dari pengguna (10 nilai untuk time series)
input_data = []
for i in range(10):
    value = st.number_input(
        f"Data ke-{i+1} (suhu dalam derajat Celsius):",
        min_value=0.0,
        max_value=100.0,
        value=0.5,
        step=0.1,
        key=f"input_{i}"
    )
    input_data.append(value)

# Tombol untuk memprediksi
if st.button("Prediksi"):
    try:
        prediction = predict(input_data)
        st.success(f"Hasil prediksi suhu: {prediction:.4f} °C")
    except Exception as e:
        st.error(f"Error saat memprediksi: {e}")

# (Opsional) Tampilkan dataset
if st.checkbox("Tampilkan beberapa baris dataset"):
    import pandas as pd
    df = pd.read_csv('jena_climate_2009_2016.csv')
    st.write(df.head())