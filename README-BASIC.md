# Metro Ağı Projesi

Bu proje, bir metro ağı simülasyonu oluşturmak ve farklı istasyonlar arasında en uygun rotaları bulmak için geliştirilmiş bir Python uygulamasıdır. Proje, metro ağındaki istasyonlar ve hatlar arasındaki bağlantıları modelleyerek, kullanıcıların en az aktarmalı ve en hızlı rotaları bulabilmesini sağlar.

## Özellikler

- İstasyonlar ve hatlar oluşturma
- İstasyonlar arası bağlantılar tanımlama
- En az aktarma gerektiren rotaları bulma (BFS algoritması ile)
- En hızlı rotaları bulma (Dijkstra algoritması ile)
- Hat değişimlerinde aktarma sürelerini hesaba katma

## Proje Yapısı

Proje aşağıdaki temel sınıflardan oluşmaktadır:

1. **Istasyon**: Her bir metro istasyonunu temsil eder.
   - İstasyon kimliği (idx)
   - İstasyon adı (ad)
   - Bağlı olduğu hat (hat)
   - Komşu istasyonlar listesi (komsular)

2. **MetroAgi**: Tüm metro ağını temsil eder ve algoritmaları barındırır.
   - İstasyon ekleme
   - Bağlantı ekleme
   - En az aktarmalı rota bulma (BFS algoritması)
   - En hızlı rota bulma (Dijkstra algoritması)

## Kullanım

```python
# Metro ağı oluşturma
metro = MetroAgi()

# İstasyonlar ekleme
metro.istasyon_ekle("K1", "Kızılay", "Kırmızı Hat")
metro.istasyon_ekle("M1", "AŞTİ", "Mavi Hat")

# Bağlantılar ekleme
metro.baglanti_ekle("K1", "K2", 4)  # Kızılay -> Ulus arası 4 dakika

# En az aktarmalı rota bulma
rota = metro.en_az_aktarma_bul("M1", "K4")
if rota:
    print("En az aktarmalı rota:", " -> ".join(i.ad for i in rota))

# En hızlı rota bulma
sonuc = metro.en_hizli_rota_bul("M1", "K4")
if sonuc:
    rota, sure = sonuc
    print(f"En hızlı rota ({sure} dakika):", " -> ".join(i.ad for i in rota))
```

## Algoritmalar

### BFS (Breadth-First Search) Algoritması
En az aktarma gerektiren rotaları bulmak için kullanılır. BFS algoritması, bir başlangıç noktasından başlayarak, aynı hatta devam eden istasyonları önceliklendirir ve en az hat değişimi gerektiren rotayı bulur.

### Dijkstra Algoritması
En hızlı rotayı bulmak için kullanılır. Her bağlantının süresini ve hat değişimlerinde oluşan ek süreleri (5 dakika) hesaba katarak, toplam seyahat süresini minimize eden rotayı bulur.

## Örnek Senaryo

Örnek metro ağımızda aşağıdaki hatlar ve istasyonlar bulunmaktadır:

- **Kırmızı Hat**: Kızılay, Ulus, Demetevler, OSB
- **Mavi Hat**: AŞTİ, Kızılay, Sıhhiye, Gar
- **Turuncu Hat**: Batıkent, Demetevler, Gar, Keçiören

Aktarma noktaları:
- Kızılay (Kırmızı Hat ve Mavi Hat arasında)
- Demetevler (Kırmızı Hat ve Turuncu Hat arasında)
- Gar (Mavi Hat ve Turuncu Hat arasında)

## Geliştirme

Projeye katkıda bulunmak için şu adımları izleyebilirsiniz:

1. Yeni istasyonlar ve hatlar ekleyin
2. Metro ağı veri yapısını genişletin
3. Yeni rota bulma algoritmaları ekleyin
4. Kullanıcı arayüzü geliştirebilirsiniz

## Gelecek Özellikler

- Gerçek zamanlı trafik verilerini entegre etme
- İstasyon çıkışları ve aktarma detaylarını ekleme
- Web arayüzü ile görselleştirme
- Farklı tercihler bazında optimizasyon (en az yürüme, en az maliyet vb.)
