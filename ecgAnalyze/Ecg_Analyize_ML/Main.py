import file_system_edf_qrs as md1
import first_file as md2
import Second_file as md3
from multiprocessing import Process
import os

def grafic_md1():
    md1.grafic()

def grafic_md2():
    md2.grafic()

def grafic_md3():
    md3.grafic()

def main():
   """ # Paralel işlemleri başlat
    p1 = Process(target=grafic_md1)
    p2 = Process(target=grafic_md2)
    p3 = Process(target=grafic_md3)

    # İşlemleri başlat
    p1.start()
    p2.start()
    p3.start()

    # İşlemlerin bitmesini bekle
    p1.join()
    p2.join()
    p3.join()"""
   for item in os.listdir('data/edf_file'):
    path = os.path.join("data/edf_file",item)
    md3.grafic(path)




if __name__ == "__main__":
    main()
