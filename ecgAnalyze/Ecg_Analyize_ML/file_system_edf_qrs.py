import pyedflib
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np


def edf_write(path):

    edf_file = pyedflib.EdfReader(path)
    kanal_sayisi = edf_file.signals_in_file
    #print(f"Kanal Sayısı: {kanal_sayisi}")
    sinyaller = edf_file.readSignal(0)
    #print(sinyaller)
    edf_file.close()
    return  sinyaller

def edf_image(path):

    edf_file = pyedflib.EdfReader(path)
    sinyal = edf_file.readSignal(0)
    edf_file.close()
    # Ses sinyalini görselleştir
    plt.plot(sinyal)
    plt.title("Ses Sinyali")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")
    plt.show()

def qrs_image(path):

    with open(path, "rb") as file:
        data = np.fromfile(file, dtype=np.int16)  # Veri türü cihazdan cihaza değişebilir


    plt.plot(data)
    plt.title("QRS Ses Sinyali")
    plt.xlabel("Zaman")
    plt.ylabel("Amplitüd")
    plt.show()


def data2voice(sinyal):
    sampling_rate = 1000

    # Sinyal veri tipi genelde float olabilir, bunu int16 formatına dönüştürmeliyiz
    sinyal = sinyal.astype(np.int16)

    # WAV dosyasına yaz
    write("cikti_sesi.wav", sampling_rate, sinyal)
    print("Ses dosyası oluşturuldu: cikti_sesi.wav")


def read_edf_channels(path):
    """
    EDF dosyasındaki tüm kanalları okur.
    """
    try:
        edf_file = pyedflib.EdfReader(path)
        kanal_sayisi = edf_file.signals_in_file
        print(f"Kanal Sayısı: {kanal_sayisi}")

        # Tüm kanalları oku
        sinyaller = [edf_file.readSignal(i) for i in range(kanal_sayisi)]
        edf_file.close()
        return sinyaller
    except Exception as e:
        print(f"EDF dosyası okunamadı: {e}")
        return None


def plot_all_channels_in_one(sinyaller, titles=None, save_path=None):
    """
    Tüm kanalları tek bir grafik üzerinde veya alt grafiklerde (subplot) gösterir.

    Parametreler:
    - sinyaller (list of lists): Her bir kanalın sinyal verileri.
    - titles (list of str, optional): Her kanal için başlıklar.
    - save_path (str, optional): Eğer belirtilirse grafik bu dosya yoluna kaydedilir.
    """
    kanal_sayisi = len(sinyaller)
    fig, axs = plt.subplots(kanal_sayisi, 1, figsize=(12, 3 * kanal_sayisi), sharex=True)

    if kanal_sayisi == 1:  # Tek kanal varsa axs bir eksen olur, listeye çeviriyoruz.
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.plot(sinyaller[i], label=f"Kanal {i + 1}")
        title = titles[i] if titles and i < len(titles) else f"Kanal {i + 1}"
        ax.set_title(title)
        ax.set_ylabel("Amplitüd")
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Zaman")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")

    plt.show()


def read_edf_channels(path):
    """
    EDF dosyasından kanalları okur. Örnekleme amacıyla basit bir mock fonksiyon.

    Parametreler:
    - path (str): EDF dosyasının yolu.

    Dönüş:
    - list of lists: Kanalların sinyal verileri.
    """
    # Bu kısım EDF okuyucu bir kütüphane kullanılarak düzenlenebilir, örneğin pyEDFlib.
    # Aşağıdaki örnek veridir.
    num_channels = 5
    signal_length = 1000  # Her kanalın sinyal uzunluğu
    return [np.sin(np.linspace(0, 10, signal_length) + i) for i in range(num_channels)]


import os
import matplotlib.pyplot as plt
import numpy as np


def plot_all_channels_in_one(sinyaller, titles=None, save_path=None):
    """
    Tüm kanalları tek bir grafik üzerinde veya alt grafiklerde (subplot) gösterir.

    Parametreler:
    - sinyaller (list of lists): Her bir kanalın sinyal verileri.
    - titles (list of str, optional): Her kanal için başlıklar.
    - save_path (str, optional): Eğer belirtilirse grafik bu dosya yoluna kaydedilir.
    """
    kanal_sayisi = len(sinyaller)
    fig, axs = plt.subplots(kanal_sayisi, 1, figsize=(12, 3 * kanal_sayisi), sharex=True)

    if kanal_sayisi == 1:  # Tek kanal varsa axs bir eksen olur, listeye çeviriyoruz.
        axs = [axs]

    for i, ax in enumerate(axs):
        ax.plot(sinyaller[i], label=f"Kanal {i + 1}")
        title = titles[i] if titles and i < len(titles) else f"Kanal {i + 1}"
        ax.set_title(title)
        ax.set_ylabel("Amplitüd")
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Zaman")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")

    plt.show()


def read_edf_channels(path):
    """
    EDF dosyasından kanalları okur. Örnekleme amacıyla basit bir mock fonksiyon.

    Parametreler:
    - path (str): EDF dosyasının yolu.

    Dönüş:
    - list of lists: Kanalların sinyal verileri.
    """
    # Bu kısım EDF okuyucu bir kütüphane kullanılarak düzenlenebilir, örneğin pyEDFlib.
    # Aşağıdaki örnek veridir.
    num_channels = 5
    signal_length = 1000  # Her kanalın sinyal uzunluğu
    return [np.sin(np.linspace(0, 10, signal_length) + i) for i in range(num_channels)]


def grafic(path):
    """
    Ana kontrol fonksiyonu. EDF dosyasından kanalları okur ve grafiklerini çizer.

    Parametreler:
    - path (str): EDF dosyasının yolu.
    """
    kanal_aciklamalari = [
        "Göbek etrafındaki elektrot 1",
        "Göbek etrafındaki elektrot 2",
        "Göbek etrafındaki elektrot 3",
        "Göbek etrafındaki elektrot 4",
        "Doğrudan fetal EKG"
    ]

    # EDF Dosyasını Oku
    sinyaller = read_edf_channels(path)
    if sinyaller:
        # Dosya adına göre grafik kaydetme
        dosya_adi = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_path = os.path.join("output", dosya_adi)

        # Çıkış dizini yoksa oluştur
        os.makedirs("output", exist_ok=True)

        # Tüm kanalları grafikle
        plot_all_channels_in_one(sinyaller, titles=kanal_aciklamalari, save_path=save_path)






