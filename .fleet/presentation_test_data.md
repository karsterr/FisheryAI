# 1. Kategori: Avcılık Üretimi (Capture Fisheries)

Modelin "Güçlü Artış" ve "Sert Düşüş" tespitini göstermek için.

### Test 1: İdeal Artış (Pozitif Senaryo)
Senaryo: Bir ülkenin balıkçılık filosu büyüyor, üretim her yıl artıyor. \
Veri: 25000, 28000, 31000, 35000, 42000 \
Konu Seçimi: 1 (Avcılık Üretimi) \
Beklenen Sonuç: "GÜÇLÜ ARTIŞ SEYRETMİŞTİR"

### Test 2: Ani Çöküş (Negatif Senaryo)
Senaryo: Aşırı avlanma veya çevre felaketi sonucu üretim çakılıyor. \
Veri: 50000, 48000, 40000, 30000, 22000 \
Konu Seçimi: 1 (Avcılık Üretimi) \
Beklenen Sonuç: "SERT DÜŞÜŞ SEYRETMİŞTİR"

### Test 3: Dalgalı / Durağan (Nötr Senaryo)
Senaryo: Sektör doyum noktasına ulaşmış, değişim az. \
Veri: 12000, 12500, 11800, 12200, 12100 \
Konu Seçimi: 1 (Avcılık Üretimi) \
Beklenen Sonuç: "DURAĞAN SEYRETMİŞTİR" (veya Kısmi Artış/Azalış)

# 2. Kategori: Yetiştiricilik (Aquaculture)
Yetiştiricilik genelde hızlı büyüyen bir sektördür, bunu test ediyoruz.

### Test 4: Sektör Patlaması (Booming)
Senaryo: Yeni balık çiftlikleri kurulmuş, üretim katlanıyor. \
Veri: 1000, 1500, 2500, 4000, 6500 \
Konu Seçimi: 2 (Yetiştiricilik) \
Beklenen Sonuç: "GÜÇLÜ ARTIŞ SEYRETMİŞTİR"

### Test 5: Başlangıç Seviyesi (Stabil)
Senaryo: Küçük ölçekli, kendi halinde bir üretim. \
Veri: 500, 520, 510, 530, 540 \
Konu Seçimi: 2 (Yetiştiricilik) \
Beklenen Sonuç: "DURAĞAN / KISMİ ARTIŞ"

# 3. Kategori: Tüketim (Consumption)
Kişi başı tüketim verileri daha küçük sayılardır (kg/kişi).

### Test 6: Sağlıklı Toplum (Artan Talep)
Senaryo: Halk daha fazla balık yemeye başlıyor. \
Veri: 12.5, 13.0, 14.2, 15.8, 18.5 \
Konu Seçimi: 3 (Tüketim) \
Beklenen Sonuç: "GÜÇLÜ ARTIŞ SEYRETMİŞTİR"

### Test 7: Değişen Alışkanlıklar (Düşüş)
Senaryo: Fiyatlar artmış, halk balıktan vazgeçiyor. \
Veri: 25.0, 24.5, 22.0, 20.0, 18.0 \
Konu Seçimi: 3 (Tüketim) \
Beklenen Sonuç: "SERT DÜŞÜŞ / AZALIŞ"

# 4. Kategori: Stok Sürdürülebilirliği (Fish Stocks) - EN ÖNEMLİSİ
Burası projenin "Çevreci" yanını gösterir. Modelin sadece sayıya değil, bağlama (context) baktığını burada kanıtlarsın.

### Test 8: Kırmızı Alarm (Kritik Seviye)
Senaryo: Stoklar biyolojik limitin altına (%60) inmiş. Tehlike çanları. \
Veri: 70, 65, 55, 48, 40 \
Konu Seçimi: 4 (Stok Sürdürülebilirliği) \
Beklenen Sonuç: "KRİTİK SEVİYEDE SEYRETMİŞTİR" \
(Not: Diğer kategorilerde buna 'Düşüş' derdi ama burada 'Kritik' demesi modelin zekasını gösterir.)

### Test 9: Riskli Düşüş (Uyarı)
Senaryo: Henüz felaket yok ama gidişat kötü. \
Veri: 90, 88, 85, 80, 75 \
Konu Seçimi: 4 (Stok Sürdürülebilirliği) \
Beklenen Sonuç: "RİSKLİ DÜŞÜŞTE / KISMİ AZALIŞ"

### Test 10: İdeal Sürdürülebilirlik
Senaryo: İyi yönetilen denizler. \
Veri: 85, 86, 88, 87, 89 \
Konu Seçimi: 4 (Stok Sürdürülebilirliği) \
Beklenen Sonuç: "SÜRDÜRÜLEBİLİR / DURAĞAN"