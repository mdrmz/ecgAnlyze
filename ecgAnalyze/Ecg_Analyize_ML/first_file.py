import numpy as np
from scipy.fft import fft
from scipy.signal import butter, lfilter
import pyedflib
import matplotlib.pyplot as plt
import file_system_edf_qrs as md



# Bandpass filtreleme fonksiyonu
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')


def bandpass_filter(data, lowcut=1.0, highcut=150.0, fs=1000.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)


# Fourier dönüşümü (Frekans domainine geçiş)
def extract_frequency_features(sinyaller):
    fft_features = []
    for sig in sinyaller:
        fft_vals = np.abs(fft(sig))
        fft_features.append(fft_vals[:len(sig) // 2])  # Genellikle pozitif frekanslar kullanılır
    return np.array(fft_features)


# Zaman domaini öznitelikleri çıkarma
def time_domain_features(sinyaller):
    mean = np.mean(sinyaller, axis=1)
    var = np.var(sinyaller, axis=1)
    max_val = np.max(sinyaller, axis=1)
    min_val = np.min(sinyaller, axis=1)

    # RMS (Root Mean Square) hesaplama
    rms = np.sqrt(np.mean(sinyaller ** 2, axis=1))

    return np.vstack([mean, var, max_val, min_val, rms]).T


# Öznitelik çıkarımı (zaman ve frekans)
def feature_extraction(sinyaller):
    time_features = time_domain_features(sinyaller)
    frequency_features = extract_frequency_features(sinyaller)

    # Tüm özellikleri birleştirme
    features = np.hstack([time_features, frequency_features])
    return features


# EDF dosyasından sinyalleri okuma
def load_edf_data(path):
    edf_file = pyedflib.EdfReader(path)
    signals = []
    for i in range(edf_file.signals_in_file):
        signals.append(edf_file.readSignal(i))
    edf_file.close()
    return np.array(signals)



def grafic(path):

    # Örnek dosya yolu
    sinyaller = load_edf_data(path)  # Dosya yolunu doğru şekilde belirtmelisiniz

    # Öznitelik çıkarımı
    features = feature_extraction(sinyaller)

    # Çıkarılan öznitelikleri yazdırma (örneğin ilk 5 örneği)
    print("Öznitelikler:", features[:])



    # Örnek veri (sinyaller ve özellikler)
    # Bu sinyaller ve özellikler, yukarıdaki kodunuzda çıkardığınız verilerdir.
    sinyaller = [np.random.randn(1000) for _ in range(5)]  # 5 kanal örneği
    features = [np.random.randn(1000) for _ in range(5)]   # 5 kanal için öznitelikler

    # Grafikleri düzgün yerleştirmek için 5x2 grid kullanarak
    plt.figure(figsize=(12, 10))

    # Kanal 1: Sinyal ve Özellikler
    plt.subplot(5, 2, 1)
    plt.plot(sinyaller[0])  # İlk kanalın sinyalini çizme
    plt.title("Sinyal (Kanal 1)")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")

    plt.subplot(5, 2, 2)
    plt.plot(features[0])  # İlk kanalın özniteliklerini çizme
    plt.title("Öznitelikler (Kanal 1)")
    plt.xlabel("Öznitelikler")
    plt.ylabel("Değer")

    # Kanal 2: Sinyal ve Özellikler
    plt.subplot(5, 2, 3)
    plt.plot(sinyaller[1])  # İkinci kanalın sinyalini çizme
    plt.title("Sinyal (Kanal 2)")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")

    plt.subplot(5, 2, 4)
    plt.plot(features[1])  # İkinci kanalın özniteliklerini çizme
    plt.title("Öznitelikler (Kanal 2)")
    plt.xlabel("Öznitelikler")
    plt.ylabel("Değer")

    # Kanal 3: Sinyal ve Özellikler
    plt.subplot(5, 2, 5)
    plt.plot(sinyaller[2])  # Üçüncü kanalın sinyalini çizme
    plt.title("Sinyal (Kanal 3)")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")

    plt.subplot(5, 2, 6)
    plt.plot(features[2])  # Üçüncü kanalın özniteliklerini çizme
    plt.title("Öznitelikler (Kanal 3)")
    plt.xlabel("Öznitelikler")
    plt.ylabel("Değer")

    # Kanal 4: Sinyal ve Özellikler
    plt.subplot(5, 2, 7)
    plt.plot(sinyaller[3])  # Dördüncü kanalın sinyalini çizme
    plt.title("Sinyal (Kanal 4)")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")

    plt.subplot(5, 2, 8)
    plt.plot(features[3])  # Dördüncü kanalın özniteliklerini çizme
    plt.title("Öznitelikler (Kanal 4)")
    plt.xlabel("Öznitelikler")
    plt.ylabel("Değer")

    # Kanal 5: Sinyal ve Özellikler
    plt.subplot(5, 2, 9)
    plt.plot(sinyaller[4])  # Beşinci kanalın sinyalini çizme
    plt.title("Sinyal (Kanal 5)")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")

    plt.subplot(5, 2, 10)
    plt.plot(features[4])  # Beşinci kanalın özniteliklerini çizme
    plt.title("Öznitelikler (Kanal 5)")
    plt.xlabel("Öznitelikler")
    plt.ylabel("Değer")

    # Düzenlemeyi uygula
    plt.tight_layout()


    # Grafikleri göster
    plt.show()

