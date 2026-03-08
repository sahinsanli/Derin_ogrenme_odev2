XOR Problemi - Yapay Sinir Ağı ile Çözüm
Bu proje, XOR mantık kapısı problemini sıfırdan yazılmış bir yapay sinir ağı ile çözmektedir. Herhangi bir makine öğrenmesi kütüphanesi kullanılmamıştır.

Ağ Mimarisi
Giriş Katmanı: 2 nöron
Gizli Katman: 2 nöron (Sigmoid aktivasyon)
Çıkış Katmanı: 1 nöron (Sigmoid aktivasyon)

XOR Doğruluk Tablosu
Giriş 1	Giriş 2	Çıkış
0	0	0
0	1	1
1	0	1
1	1	0

Kullanılan Yöntemler
İleri Besleme (Forward Propagation)
Geri Yayılım (Backpropagation)
Gradient Descent ile ağırlık güncelleme
MSE (Ortalama Karesel Hata)

Gereksinimler
pip install numpy

Çalıştırma
python odev.py
