import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

data = {
    'bedrooms': [3, 4, 2, 5],
    'bathrooms': [2, 3, 1, 4],
    'sqft': [1500, 2000, 800, 2500],
    'location': [1, 2, 1, 2],
    'price': [300000, 450000, 200000, 500000]
}

df = pd.DataFrame(data)

x = df[['bedrooms', 'bathrooms', 'sqft', 'location']]
y = df[['price']]

model = Sequential()

model.add(Dense(units=64, activation='relu', input_shape=(4,)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x, y, epochs=100, batch_size=1)

sample_input = np.array([[3, 2, 1500, 1]])
predicted_price = model.predict(sample_input)

print(f"Predicted price for the given house is: ${predicted_price[0][0]:,.2f}")