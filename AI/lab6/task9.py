import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

dataset = np.sin(np.arange(500)) + np.random.rand(500)*0.4
dataset = dataset.reshape(len(dataset),1)

sca = MinMaxScaler()
dataset = sca.fit_transform(dataset)

def prepare_data(dataset, start_index, end_index, history_size):
    X, y = [], []
    for i in range(start_index + history_size, end_index):
        X.append(dataset[i - history_size:i])
        y.append(dataset[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape(-1, 1)


train_split = int(len(dataset) * 0.7)
past_history = 10

X_train, y_train = prepare_data(dataset, 0, train_split, past_history)
X_test, y_test = prepare_data(dataset, train_split, len(dataset), past_history)

model = keras.Sequential([
    keras.layers.LSTM(8, input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dense(y_train.shape[1])
])

model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(learning_rate=0.001)
)

model.summary()

history = model.fit(
  X_train, y_train,
  epochs=70,
  batch_size=4,
  validation_split=0.1,
  verbose=1,
  shuffle=False
)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))

plt.plot(np.arange(0, len(y_train)), y_train, 'b', label="Training Data (History)")

plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    y_test,
    'k',
    markersize=3,
    label="Actual Test Data"
)

plt.plot(
    np.arange(len(y_train), len(y_train) + len(y_test)),
    y_pred,
    'r--',
    label="Predicted Data"
)

plt.ylabel('Value', fontsize=12)
plt.xlabel('Step', fontsize=12)
plt.title("Model Prediction vs Actual Data", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc="best", fontsize=10)
plt.tight_layout()
plt.show()


