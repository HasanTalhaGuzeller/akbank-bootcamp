# DL-ALT: GNN Destekli Metro Güzergah Optimizasyonu

## Proje Açıklaması

DL-ALT algoritması, metro ağlarında en hızlı rotayı belirlemek için *Derin Öğrenme ve Landmark Tabanlı ALT (A** Landmark Heuristic) arama*\* yöntemlerini birleştirir. Grafik Sinir Ağları (GNN) kullanılarak metro istasyonları arasında en önemli noktalar (landmark'lar) seçilir ve A\* algoritmasının sezgisel fonksiyonu bu landmark'lar üzerinden optimize edilir.

## Kullanılan Teknolojiler

- **Python** (Ana programlama dili)
- **PyTorch & PyTorch Geometric** (GNN modeli için)
- **NetworkX & heapq** (Klasik grafik algoritmaları için)

## Kurulum

Projeyi çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

```bash
# Gerekli kütüphaneleri yükleyin
pip install torch torch-geometric numpy

# Ana Python dosyasını çalıştırın
python metro_gnn.py
```

## Algoritma Açıklaması

DL-ALT algoritması aşağıdaki aşamalardan oluşur:

1. **Grafın Hazırlanması:** Metro istasyonları ve bağlantıları bir grafik yapısında modellenir.
2. **GNN ile Landmark Seçimi:** Grafik Sinir Ağları (GNN) kullanılarak, yüksek öneme sahip istasyonlar belirlenir.
3. **DL-ALT Arama:** Seçilen landmark'lar, A\* algoritmasının sezgisel fonksiyonunu güçlendirmek için kullanılır.
4. **Karşılaştırmalı Performans Testleri:** Klasik Dijkstra, A\* ve ALT algoritmaları ile karşılaştırılır.

## Performans Karşılaştırmaları

| Algoritma  | Ortalama Hesaplama Süresi | Hafıza Kullanımı | Doğruluk |
| ---------- | ------------------------- | ---------------- | -------- |
| Dijkstra   | Yüksek                    | Orta             | 100%     |
| A\*        | Orta                      | Orta             | 100%     |
| ALT        | Düşük                     | Orta             | 100%     |
| **DL-ALT** | **Çok Düşük**             | **Düşük**        | **100%** |

**DL-ALT**, klasik algoritmalara göre daha düşük hesaplama süresi ve hafıza kullanımı ile optimum çözümler sunmaktadır.

## Kullanım Örneği

```python
from metro_gnn import MetroWithDLALT

dl_metro = MetroWithDLALT()
dl_metro.train_landmark_selector(epochs=50)
rota, sure = dl_metro.dl_alt_search("M1", "K4")
print(f"En hızlı rota ({sure} dakika):", " -> ".join(i.ad for i in rota))
```

## Sonuç

DL-ALT algoritması, metro gibi büyük ölçekli ağlarda **hızlı, verimli ve akıllı** güzergah hesaplaması sağlar. GNN kullanımı sayesinde, istasyonların trafik durumuna göre dinamik olarak en uygun landmark'lar belirlenebilir.

**Gelecekteki İyileştirmeler:**

- Gerçek zamanlı trafik yoğunluğu entegrasyonu
- Farklı ulaşım modlarıyla entegrasyon (otobüs, tramvay, vb.)
- Daha büyük metropol alanlarında test edilmesi

---

Bu proje, **şehir içi ulaşımı optimize etmek için yapay zeka ve grafik algoritmalarının nasıl entegre edilebileceğini** göstermektedir.


