import tensorflow as tf
from tensorflow.keras.models import Sequential   # cetakan untuk menumpuk layer secara berurutan
from tensorflow.keras.layers import Dense        # layer "fully connected" (semua neuron saling terhubung)
import pandas as pd                               # untuk membuat & mengolah data dalam bentuk tabel
import numpy as np                                # untuk operasi angka/array

# =========================================================
# 1. DATA — contoh 4 rumah yang akan dipakai untuk melatih model
# =========================================================
data = {
    'bedrooms': [3, 4, 2, 5],          # jumlah kamar tidur
    'bathrooms': [2, 3, 1, 4],         # jumlah kamar mandi
    'sqft': [1500, 2000, 800, 2500],   # luas rumah (dalam sqft)
    'location': [1, 2, 1, 2],          # kode lokasi (1 atau 2)
    'price': [300000, 450000, 200000, 500000]  # harga rumah -> ini target/jawaban yang ingin dipelajari model
}

# Ubah dictionary di atas jadi tabel rapi (baris & kolom)
df = pd.DataFrame(data)

# =========================================================
# 2. PEMISAHAN FITUR (X) DAN TARGET (Y)
# =========================================================
# x = "soal" yang dilihat model (4 kolom input)
x = df[['bedrooms', 'bathrooms', 'sqft', 'location']]

# y = "jawaban benar" yang ingin dipelajari model untuk diprediksi
y = df[['price']]

# =========================================================
# 3. ARSITEKTUR MODEL — menyusun jaringan saraf tiruan
# =========================================================
model = Sequential()  # model kosong, siap diisi layer satu per satu

# Hidden layer 1: 64 neuron
# input_shape=(4,) artinya layer ini menerima 4 angka sebagai input (sesuai jumlah kolom di x)
# activation='relu' membuat model bisa mengenali pola yang tidak sekadar garis lurus
model.add(Dense(units=64, activation='relu', input_shape=(4,)))

# Hidden layer 2: 32 neuron
# Menerima 64 angka dari layer sebelumnya, lalu meringkasnya jadi representasi baru
model.add(Dense(units=32, activation='relu'))

# Output layer: 1 neuron, tanpa activation
# Hanya butuh 1 angka keluar (prediksi harga), dan angkanya harus bebas (tidak dibatasi rentang)
model.add(Dense(units=1))

# =========================================================
# 4. COMPILE — menentukan cara model belajar
# =========================================================
# loss='mean_squared_error' -> mengukur seberapa salah prediksi (rata-rata dari (prediksi - harga_asli)^2)
# optimizer='adam' -> algoritma yang mengatur seberapa besar & ke arah mana bobot diubah agar loss makin kecil
model.compile(optimizer='adam', loss='mean_squared_error')

# =========================================================
# 5. TRAINING — proses belajar model
# =========================================================
# epochs=100   -> model "membaca ulang" seluruh data training sebanyak 100 kali
# batch_size=1 -> bobot di-update setiap kali model selesai memproses 1 baris data
model.fit(x, y, epochs=100, batch_size=1)

# =========================================================
# 6. PREDIKSI — menguji model dengan data rumah baru
# =========================================================
# Data baru: 3 kamar tidur, 2 kamar mandi, 1500 sqft, lokasi 1
# Urutan angka HARUS sama persis dengan urutan kolom di x
sample_input = np.array([[3, 2, 1500, 1]])

# Mengalirkan data baru lewat model yang sudah dilatih (tanpa update bobot lagi)
predicted_price = model.predict(sample_input)

# =========================================================
# 7. MENAMPILKAN HASIL
# =========================================================
# predicted_price berbentuk array 2 dimensi, jadi ambil angkanya dengan [0][0]
# :,.2f -> format angka dengan pemisah ribuan dan 2 angka di belakang koma
print(f"Prediksi harga untuk rumah tersebut adalah: Rp{predicted_price[0][0]:,.2f}")