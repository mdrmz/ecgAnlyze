import librosa
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
from scipy.interpolate import interp1d
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


# MFCC Özellik Çıkarımı
def extract_mfcc_features(signal, fs=1000, coefficients=[13, 26, 39]):
    """MFCC özniteliklerini çıkarır ve interpolasyonla uyarlar"""
    n_mfcc = max(coefficients) + 1
    mfcc = librosa.feature.mfcc(y=signal.astype(np.float32), sr=fs, n_mfcc=n_mfcc)
    selected_indices = [c - 1 for c in coefficients]
    mfcc_selected = mfcc[selected_indices]

    # Interpolasyon
    n_frames = mfcc_selected.shape[1]
    original_length = len(signal)
    x = np.linspace(0, original_length, num=n_frames)
    f = interp1d(x, mfcc_selected, axis=1, bounds_error=False, fill_value='extrapolate')
    x_new = np.arange(original_length)
    return f(x_new).T


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


# Regresyon Modelleri
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
    # Veri yükleme
    edf_file = 'data/edf_file/r01.edf'
    f = pyedflib.EdfReader(edf_file)
    signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    # Sinyal ayrıştırma
    abdominal_signals = signals[:4]
    fetal_signal = signals[4]

    # Ön işleme
    filtered_signals = [bandpass_filter(s) for s in abdominal_signals]
    fetal_signal_filtered = bandpass_filter(fetal_signal)

    filtered_signals = [power_line_interference_removal(s) for s in filtered_signals]
    fetal_signal_filtered = power_line_interference_removal(fetal_signal_filtered)

    filtered_signals = [baseline_drift_correction(s) for s in filtered_signals]
    fetal_signal_filtered = baseline_drift_correction(fetal_signal_filtered)

    filtered_signals = [wavelet_denoising(s) for s in filtered_signals]
    fetal_signal_filtered = wavelet_denoising(fetal_signal_filtered)

    # Normalizasyon
    normalized_signals = [normalize_signal(s, 'minmax') for s in filtered_signals]
    fetal_signal_normalized = normalize_signal(fetal_signal_filtered, 'minmax')

    # MFCC Özellik Çıkarımı (13, 26, 39)
    mfcc_features = []
    for signal in normalized_signals:
        mfcc = extract_mfcc_features(signal, fs=1000)
        mfcc_features.append(mfcc)

    # Özellik Birleştirme
    abdominal_features = np.array(normalized_signals).T
    mfcc_features = np.hstack(mfcc_features)
    X = np.concatenate([abdominal_features, mfcc_features], axis=1)
    y = fetal_signal_normalized

    # Veri bölümleme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model eğitim ve test
    models = {
        'Random Forest': random_forest_regressor,
        'SVM': svm_regressor,
        'Gradient Boosting': gradient_boosting_regressor
    }

    results = {}
    for name, model_func in models.items():
        predictions = model_func(X_train, y_train, X_test)
        results[name] = {
            'MSE': mean_squared_error(y_test, predictions),
            'R2': r2_score(y_test, predictions)
        }

    # Sonuçları görüntüle
    for model_name, metrics in results.items():
        print(f"\n{model_name} Performansı:")
        print(f"Mean Squared Error: {metrics['MSE']:.4f}")
        print(f"R² Score: {metrics['R2']:.4f}")

    # Görselleştirme
    plt.figure(figsize=(15, 5))
    for i, (name, pred) in enumerate(zip(models.keys(), [random_forest_regressor(X_train, y_train, X_test),
                                                         svm_regressor(X_train, y_train, X_test),
                                                         gradient_boosting_regressor(X_train, y_train, X_test)])):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'{name} Tahminleri')
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahminler')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()