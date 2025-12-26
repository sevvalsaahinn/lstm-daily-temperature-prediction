import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from model import build_lstm_model

import os

os.makedirs("results", exist_ok=True)


SEQ_LENGTH = 7
EPOCHS = 20
BATCH_SIZE = 16


df = pd.read_csv(
    "data/temperatures.csv",
    header=None,
    engine="python",
    on_bad_lines="skip"
)


df = df[df[0].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]

df.columns = ["date", "temp"]

df["date"] = pd.to_datetime(df["date"])
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")

df = df.dropna().reset_index(drop=True)

values = df["temp"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)


X, y = [], []

for i in range(len(scaled) - SEQ_LENGTH):
    X.append(scaled[i:i + SEQ_LENGTH])
    y.append(scaled[i + SEQ_LENGTH])

X = np.array(X)
y = np.array(y)


split_index = int(0.8 * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


model = build_lstm_model((SEQ_LENGTH, 1))

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)


plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig("results/loss.png")
plt.close()


predictions = model.predict(X_test)

predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
print("RMSE:", rmse)


plt.figure()
plt.plot(y_test_inv, label="Gerçek")
plt.plot(predictions_inv, label="Tahmin")
plt.xlabel("Zaman")
plt.ylabel("Sıcaklık (°C)")
plt.title("Gerçek vs Tahmin")
plt.legend()
plt.savefig("results/prediction.png")
plt.close()

model.save("results/lstm_model.keras")

