import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

# Data contoh rumah (4 baris data latih)
data = {
    'bedrooms': [3, 4, 2, 5],       # jumlah kamar tidur
    'bathrooms': [2, 3, 1, 4],      # jumlah kamar mandi
    'sqft': [1500, 2000, 800, 2500],  # luas rumah (dalam sqft)
    'location': [1, 2, 1, 2],       # kode lokasi
    'price': [300000, 450000, 200000, 500000]  # harga rumah (target)
}

df = pd.DataFrame(data)

# Memisahkan fitur (input) dan target (output) yang ingin diprediksi
x = df[['bedrooms', 'bathrooms', 'sqft', 'location']]  # fitur input
y = df[['price']]                                        # target/label (harga)

# Membangun arsitektur model jaringan saraf tiruan
model = Sequential()

model.add(Dense(units=64, activation='relu', input_shape=(4,)))  # hidden layer 1 - 64 neuron
model.add(Dense(units=32, activation='relu'))                    # hidden layer 2 - 32 neuron
model.add(Dense(units=1))                                        # output layer - 1 neuron (harga)

# Menentukan cara model belajar
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model dengan data yang ada
model.fit(x, y, epochs=30, batch_size=1)

# Data baru untuk diuji/diprediksi
sample_input = np.array([[3, 2, 1500, 1]])
predicted_price = model.predict(sample_input)

print(f"Prediksi harga untuk rumah tersebut adalah: Rp{predicted_price[0][0]:,.2f}")