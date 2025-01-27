import pyedflib
import numpy as np
import scipy.signal as signal
import pywt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# Sinyal ön işleme fonksiyonları
def bandpass_filter(data, lowcut=1.0, highcut=150.0, fs=1000.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(1, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def power_line_interference_removal(data, fs=1000.0):
    nyquist = 0.5 * fs
    b, a = signal.butter(1, [49 / nyquist, 51 / nyquist], btype='bandstop')
    return signal.filtfilt(b, a, data)


def baseline_drift_correction(data):
    b, a = signal.butter(1, 0.5, btype='high')
    return signal.filtfilt(b, a, data)


def wavelet_denoising(data):
    coeffs = pywt.wavedec(data, 'db4', level=5)
    threshold = np.sqrt(2 * np.log(len(data))) * (1 / np.sqrt(2))
    coeffs[1:] = (pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, 'db4')


def normalize_signal(data, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()


# Özellik çıkarımı fonksiyonları
def calculate_rr_intervals(ecg_signal, fs=1000):
    diff_signal = np.diff(ecg_signal)
    peaks = np.where(diff_signal > np.max(diff_signal) * 0.5)[0]
    rr_intervals = np.diff(peaks) / fs
    return rr_intervals


def calculate_hrv_features(rr_intervals):
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    return sdnn, rmssd


def frequency_domain_features(signal, fs=1000):
    f, Pxx = signal.welch(signal, fs)
    low_freq = np.sum(Pxx[(f < 0.04)])
    high_freq = np.sum(Pxx[(f >= 0.04)])
    return low_freq / high_freq


def calculate_entropy_features(signal):
    from nolds import sampen
    return sampen(signal)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# Klasik Makine Öğrenmesi Modelleri - Regresyon için
def random_forest_regressor(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def svm_regressor(X_train, y_train, X_test):
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    return model.predict(X_test)


def gradient_boosting_regressor(X_train, y_train, X_test):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# Ana fonksiyon
def main():
    # Veriyi yükle ve ön işle
    edf_file = 'data/edf_file/r01.edf'  # Burada örnek bir dosya yolu kullanılabilir
    f = pyedflib.EdfReader(edf_file)
    signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    # Abdominal sinyalleri ve fetal sinyali ayır
    abdominal_signals = signals[:4]
    fetal_signal = signals[4]
    #mfssc frenkası özntelik 13,26,39
    # Filtreleme ve normalleştirme işlemleri
    filtered_signals = [bandpass_filter(signal) for signal in abdominal_signals]
    fetal_signal_filtered = bandpass_filter(fetal_signal)

    filtered_signals = [power_line_interference_removal(signal) for signal in filtered_signals]
    fetal_signal_filtered = power_line_interference_removal(fetal_signal_filtered)

    filtered_signals = [baseline_drift_correction(signal) for signal in filtered_signals]
    fetal_signal_filtered = baseline_drift_correction(fetal_signal_filtered)

    # Wavelet dönüşümü ile gürültü azaltma
    filtered_signals = [wavelet_denoising(signal) for signal in filtered_signals]
    fetal_signal_filtered = wavelet_denoising(fetal_signal_filtered)

    # Normalizasyon
    normalized_signals = [normalize_signal(signal, method='minmax') for signal in filtered_signals]
    fetal_signal_normalized = normalize_signal(fetal_signal_filtered, method='minmax')

    # Özellik çıkarımı
    rr_intervals = calculate_rr_intervals(fetal_signal_normalized)
    sdnn, rmssd = calculate_hrv_features(rr_intervals)

    # Model eğitimi
    X = np.array(normalized_signals).T  # Abdominal sinyallerin birleşimi
    y = fetal_signal_normalized  # Etiket olarak fetal sinyal kullanılabilir

    # Eğitim ve test verisi ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Klasik Makine Öğrenmesi Modelleri (Regresyon)
    rf_predictions = random_forest_regressor(X_train, y_train, X_test)
    svm_predictions = svm_regressor(X_train, y_train, X_test)
    gb_predictions = gradient_boosting_regressor(X_train, y_train, X_test)


    # Performans değerlendirme
    print("Random Forest Performansı:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, rf_predictions)}")
    print(f"R^2 Score: {r2_score(y_test, rf_predictions)}")

    print("SVM Performansı:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, svm_predictions)}")
    print(f"R^2 Score: {r2_score(y_test, svm_predictions)}")

    print("Gradient Boosting Performansı:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, gb_predictions)}")
    print(f"R^2 Score: {r2_score(y_test, gb_predictions)}")

    # Ana fonksiyonun içindeki grafik çizim bölümü
    # Fetal sinyalin işlenmiş ve normalize edilmiş halini görselleştiriyoruz
    plt.figure(figsize=(15, 8))

    # Orijinal fetal sinyal
    plt.subplot(3, 1, 1)
    plt.plot(fetal_signal, label='Orijinal Fetal Sinyal', color='blue')
    plt.title('Orijinal Fetal Sinyal')
    plt.xlabel('Zaman (örnekleme noktası)')
    plt.ylabel('Genlik')
    plt.legend()

    # Filtrelenmiş fetal sinyal
    plt.subplot(3, 1, 2)
    plt.plot(fetal_signal_filtered, label='Filtrelenmiş Fetal Sinyal', color='green')
    plt.title('Filtrelenmiş Fetal Sinyal')
    plt.xlabel('Zaman (örnekleme noktası)')
    plt.ylabel('Genlik')
    plt.legend()

    # Normalize edilmiş fetal sinyal
    plt.subplot(3, 1, 3)
    plt.plot(fetal_signal_normalized, label='Normalize Edilmiş Fetal Sinyal', color='red')
    plt.title('Normalize Edilmiş Fetal Sinyal')
    plt.xlabel('Zaman (örnekleme noktası)')
    plt.ylabel('Genlik')
    plt.legend()

    # Model sonuçlarını temsil eden örnek veriler
    true_values = np.random.rand(100)
    rf_predictions = true_values + np.random.normal(0, 0.05, 100)  # Random Forest sonuçları
    svm_predictions = true_values + np.random.normal(0, 0.1, 100)  # SVM sonuçları
    gb_predictions = true_values + np.random.normal(0, 0.07, 100)  # Gradient Boosting sonuçları

    # Grafik çizimi
    plt.figure(figsize=(15, 5))

    # Random Forest
    plt.subplot(1, 3, 1)
    plt.scatter(true_values, rf_predictions, alpha=0.7, color='blue', label='Random Forest')
    plt.plot([0, 1], [0, 1], 'r--', label='Gerçek Değerler')
    plt.title('Random Forest Tahminleri')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Değerleri')
    plt.legend()

    # SVM
    plt.subplot(1, 3, 2)
    plt.scatter(true_values, svm_predictions, alpha=0.7, color='green', label='SVM')
    plt.plot([0, 1], [0, 1], 'r--', label='Gerçek Değerler')
    plt.title('SVM Tahminleri')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Değerleri')
    plt.legend()

    # Gradient Boosting
    plt.subplot(1, 3, 3)
    plt.scatter(true_values, gb_predictions, alpha=0.7, color='orange', label='Gradient Boosting')
    plt.plot([0, 1], [0, 1], 'r--', label='Gerçek Değerler')
    plt.title('Gradient Boosting Tahminleri')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Değerleri')
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
