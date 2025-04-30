
import numpy as np
import tensorflow as tf
import joblib

# Muat scaler
scaler = joblib.load('scaler.pkl')

# Muat model TFLite
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fungsi untuk preprocessing dan prediksi
def predict(data):
    data_scaled = scaler.transform(np.array(data).reshape(-1, 1))
    data_scaled = data_scaled.reshape(1, 10, 1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], data_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

if __name__ == "__main__":
    sample_data = [0.5] * 10
    prediction = predict(sample_data)
    print(f"Prediksi: {prediction}")
