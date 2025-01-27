import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt

# Güç hattı girişimi filtreleme (50 Hz)
def power_line_interference_filter(signal, fs, freq=50):
    nyquist = 0.5 * fs
    low = (freq - 1) / nyquist
    high = (freq + 1) / nyquist
    b, a = butter(1, [low, high], btype='bandstop')
    return filtfilt(b, a, signal)

# Düşük frekans gürültüsünü bastırma
def low_pass_filter(signal, fs, cutoff=0.5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

# Maternal kalp sinyalini çıkarmak için basit bir bandpass filtresi
def maternal_ecg_removal(signal, fs):
    # Maternal kalp sinyali genellikle 1-2 Hz arasında
    low = 0.5 / (0.5 * fs)
    high = 3.0 / (0.5 * fs)
    b, a = butter(1, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)

# EDF dosyasından veri okuma
def read_edf_file(file_path):
    with pyedflib.EdfReader(file_path) as f:
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        fs = f.getSampleFrequency(0)  # Örnekleme frekansı
        signals = np.array([f.readSignal(i) for i in range(n)])
    return signals, signal_labels, fs

# SNR hesaplama
def compute_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# Maternal ve fetal sinyal genlik analiz
def analyze_amplitude_relationship(fetal_signal, maternal_signal):
    fetal_amplitude = np.max(np.abs(fetal_signal))
    maternal_amplitude = np.max(np.abs(maternal_signal))
    return fetal_amplitude / maternal_amplitude
from scipy.signal import find_peaks

# QRS kompleksi tespiti
def detect_qrs_complex(signal, fs):
    # Basit bir QRS tespiti için, R dalgalarını bulma
    threshold = np.max(signal) * 0.6  # Eşik değeri
    peaks, _ = find_peaks(signal, height=threshold, distance=fs/2)  # R dalgaları
    return peaks
# Fetal kalp hızı (FHR) hesaplama
def calculate_fhr(qrs_peaks, fs):
    rr_intervals = np.diff(qrs_peaks) / fs * 1000  # ms cinsinden
    fhr = 60 / (rr_intervals / 1000)  # BPM cinsinden
    return fhr
import matplotlib.pyplot as plt

# Sinyal ve QRS tespitini görselleştirme
def plot_signal_with_qrs(signal, qrs_peaks, fs):
    plt.plot(signal)
    plt.plot(qrs_peaks, signal[qrs_peaks], 'ro', label='QRS Complex')
    plt.title("Fetal ECG Signal with Detected QRS Complexes")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

# FHR'yi görselleştirme
def plot_fhr(fhr):
    plt.plot(fhr)
    plt.title("Fetal Heart Rate (FHR) over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.show()
# Dosya yolunu girin
file_path = 'data/edf_file/r01.edf'

# EDF dosyasını okuyun
signals, signal_labels, fs = read_edf_file(file_path)

# Fetal kalp sinyalini elde edin (Örneğin, 0. sinyal fetal sinyal olabilir)
fetal_signal = signals[0]

# Sinyal ön işleme
fetal_signal_filtered = power_line_interference_filter(fetal_signal, fs)
fetal_signal_filtered = low_pass_filter(fetal_signal_filtered, fs)
maternal_signal = maternal_ecg_removal(fetal_signal_filtered, fs)

# QRS tespiti
qrs_peaks = detect_qrs_complex(fetal_signal_filtered, fs)

# Fetal kalp hızı analizi
fhr = calculate_fhr(qrs_peaks, fs)

# Sonuçları görselleştirme
plot_signal_with_qrs(fetal_signal_filtered, qrs_peaks, fs)
plot_fhr(fhr)
