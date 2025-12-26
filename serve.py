import gradio as gr
import os
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model_path = os.path.join("results", "lstm_model.keras")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

print("Model yükleniyor...")
model = load_model(model_path)
print("Model başarıyla yüklendi.")


print("Scaler hazırlanıyor...")

df = pd.read_csv(
    "data/temperatures.csv",
    header=None,
    engine="python",
    on_bad_lines="skip"
)

df = df[df[0].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
df.columns = ["date", "temp"]
df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
df = df.dropna()

scaler = MinMaxScaler()
scaler.fit(df["temp"].values.reshape(-1, 1))

print("Scaler hazır.")

SEQ_LENGTH = 7


def process_input(input_text):
    try:
        if not input_text or not input_text.strip():
            return "Hata: Lütfen virgülle ayrılmış 7 sayı girin."

        values = [float(x.strip()) for x in input_text.split(",")]

        if len(values) != SEQ_LENGTH:
            return f"Hata: {SEQ_LENGTH} adet sayı bekleniyor."

        x = np.array(values).reshape(-1, 1)
        x_scaled = scaler.transform(x)
        x_scaled = x_scaled.reshape(1, SEQ_LENGTH, 1)

        prediction = model.predict(x_scaled)
        prediction = scaler.inverse_transform(prediction)

        return f"Tahmin edilen sıcaklık: {prediction[0][0]:.2f} °C"

    except Exception as e:
        return f"Hata: {str(e)}"


interface = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Son 7 günün sıcaklıklarını girin (örn: 10.2,11.0,12.3,13.1,12.7,11.8,12.0)"
    ),
    outputs=gr.Textbox(label="Çıktı"),
    title="Daily Temperature Prediction (LSTM)",
    description="Son 7 günün minimum sıcaklık değerlerini girerek bir sonraki günün tahminini elde edebilirsiniz.",
    examples=[
        ["10,11,12,13,12,11,12"],
        ["5,6,7,8,9,10,11"],
        ["14,15,16,17,18,19,20"]
    ],
)

if __name__ == "__main__":
    print("\nGradio arayüzü başlatılıyor...")
    print("Tarayıcıda açılmazsa terminaldeki URL'yi kullanın.\n")

    interface.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860
    )
