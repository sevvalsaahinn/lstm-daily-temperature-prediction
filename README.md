# LSTM Tabanlı Günlük Sıcaklık Tahmini

Bu proje, Melbourne şehrine ait günlük minimum sıcaklık verileri kullanılarak
LSTM (Long Short-Term Memory) tabanlı bir derin öğrenme modeli ile
zaman serisi tahmini gerçekleştirmeyi amaçlamaktadır.

Proje kapsamında, geçmiş günlere ait sıcaklık değerlerinden yararlanılarak
bir sonraki günün minimum sıcaklık değeri tahmin edilmiştir.
Model eğitimi, değerlendirilmesi ve kullanıcı etkileşimli bir arayüz
üzerinden servis edilmesi adımları bütüncül olarak ele alınmıştır.

---

## Kullanılan Veri Seti

- **Veri Seti Adı:** Daily Minimum Temperatures in Melbourne  
- **Zaman Aralığı:** 1981 – 1990  
- **Kaynak:**  
  https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne

Veri seti, günlük minimum sıcaklık değerlerinden oluşan bir zaman serisi
yapısına sahiptir.

---

## Proje Yapısı

```
├── data/
│ ├── temperatures.csv
│ └── README.md
├── results/
│ ├── loss.png
│ ├── prediction.png
│ ├── lstm_model.keras
│ └── README.md
├── model.py
├── train.py
├── serve.py
├── report.pdf
├── README.md
└── .gitignore

```

### Dosya Açıklamaları

- **model.py**  
  LSTM tabanlı derin öğrenme modelinin mimarisini içerir.

- **train.py**  
  Veri ön işleme, model eğitimi ve performans değerlendirme işlemlerini gerçekleştirir.
  Eğitim ve tahmin çıktıları `results/` klasörüne kaydedilir.

- **serve.py**  
  Eğitilen modelin Gradio kullanılarak web tabanlı bir kullanıcı arayüzü
  üzerinden servis edilmesini sağlar.

- **data/**  
  Projede kullanılan veri setini içerir.

- **results/**  
  Model eğitim sürecine ait kayıp grafiği, tahmin sonuçları ve
  eğitilmiş model dosyasını içerir.

- **report.pdf**  
  Projeye ait detaylı akademik raporu içerir.

---

## Model Eğitimi

Modeli eğitmek için aşağıdaki komut çalıştırılmalıdır:

```bash
python train.py
```

Bu işlem sonucunda:

- Eğitim ve doğrulama kayıp grafiği (`loss.png`)
- Gerçek ve tahmin edilen değerlerin karşılaştırıldığı grafik (`prediction.png`)
- Eğitilmiş model dosyası (`lstm_model.keras`)

`results/` klasörüne kaydedilir.

---

## Model Servisleme

Eğitilen modeli kullanıcı arayüzü üzerinden çalıştırmak için aşağıdaki komut kullanılmalıdır:

```bash
python serve.py
```
Bu komut çalıştırıldığında **Gradio tabanlı bir web arayüzü** açılır.  
Kullanıcılar, son **7 güne ait minimum sıcaklık değerlerini** girerek  
bir sonraki günün sıcaklık tahminini elde edebilir.

---

## Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Gradio

