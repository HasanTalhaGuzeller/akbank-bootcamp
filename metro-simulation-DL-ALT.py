import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import heapq
from typing import Dict, List,Tuple, Optional
from collections import defaultdict, deque


class GNNLandmarkSelector(nn.Module):
    """Landmark seçimi için Grafik Sinir Ağı modeli"""
    def __init__(self, input_dim, hidden_dim):
        super(GNNLandmarkSelector, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)  # Her düğüm için önem skoru
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # 0-1 arasında önem skoru
    
class Istasyon:
    def __init__(self, idx: str, ad: str, hat: str):
        self.idx = idx
        self.ad = ad
        self.hat = hat
        self.komsular: List[Tuple['Istasyon', int]] = []  # (istasyon, süre) tuple'ları

    def komsu_ekle(self, istasyon: 'Istasyon', sure: int):
        self.komsular.append((istasyon, sure))

class MetroAgi:
    def __init__(self):
        self.istasyonlar: Dict[str, Istasyon] = {}
        self.hatlar: Dict[str, List[Istasyon]] = defaultdict(list)

    def istasyon_ekle(self, idx: str, ad: str, hat: str) -> None:
        if idx not in self.istasyonlar:  # Burada bir hata vardı, "id" yerine "idx" olmalı
            istasyon = Istasyon(idx, ad, hat)
            self.istasyonlar[idx] = istasyon
            self.hatlar[hat].append(istasyon)

    def baglanti_ekle(self, istasyon1_id: str, istasyon2_id: str, sure: int) -> None:
        istasyon1 = self.istasyonlar[istasyon1_id]
        istasyon2 = self.istasyonlar[istasyon2_id]
        istasyon1.komsu_ekle(istasyon2, sure)
        istasyon2.komsu_ekle(istasyon1, sure)
    
    def en_az_aktarma_bul(self, baslangic_id: str, hedef_id: str) -> Optional[List[Istasyon]]:
        """BFS algoritması kullanarak en az aktarmalı rotayı bulur"""
        if baslangic_id not in self.istasyonlar or hedef_id not in self.istasyonlar:
            return None
        
        baslangic = self.istasyonlar[baslangic_id]
        hedef = self.istasyonlar[hedef_id]
        
        # BFS için kuyruk ve ziyaret edilen istasyonlar
        kuyruk = deque([(baslangic, [baslangic])])  # (istasyon, rota) tuple'ları
        ziyaret_edildi = {baslangic}
        
        while kuyruk:
            current, rota = kuyruk.popleft()
            
            # Hedef istasyona ulaşıldıysa rotayı döndür
            if current == hedef:
                return rota
            
            # Tüm komşuları kontrol et
            for komsu, _ in current.komsular:
                if komsu not in ziyaret_edildi:
                    # Hat değişikliği varsa, öncelikle aynı hatta olan istasyonları tercih et
                    yeni_rota = rota + [komsu]
                    ziyaret_edildi.add(komsu)
                    
                    # Aynı hatta devam eden istasyonları öncelikle kontrol et
                    if current.hat == komsu.hat:
                        kuyruk.appendleft((komsu, yeni_rota))  # Öncelikli olarak kuyruğun başına ekle
                    else:
                        kuyruk.append((komsu, yeni_rota))  # Aktarma gerektiren istasyonları sona ekle
        
        # Yol bulunamadıysa None döndür
        return None

    def en_hizli_rota_bul(self, baslangic_id: str, hedef_id: str) -> Optional[Tuple[List[Istasyon], int]]:
        """Dijkstra algoritması kullanarak en hızlı rotayı bulur"""
        if baslangic_id not in self.istasyonlar or hedef_id not in self.istasyonlar:
            return None

        baslangic = self.istasyonlar[baslangic_id]
        hedef = self.istasyonlar[hedef_id]
        
        # Öncelik kuyruğu: (toplam_süre, istasyon_id, istasyon, rota)
        # istasyon_id, aynı istasyona farklı sürelerle ulaşıldığında çakışmaları önlemek için eklenmiştir
        pq = [(0, id(baslangic), baslangic, [baslangic])]
        ziyaret_edildi = set()
        
        while pq:
            toplam_sure, _, current, rota = heapq.heappop(pq)
            
            # İstasyon daha önce daha kısa bir yoldan ziyaret edilmişse atla
            if current in ziyaret_edildi:
                continue
                
            # Hedef istasyona ulaşıldıysa rotayı ve toplam süreyi döndür
            if current == hedef:
                return (rota, toplam_sure)
            
            # İstasyonu ziyaret edildi olarak işaretle
            ziyaret_edildi.add(current)
            
            # Tüm komşuları kontrol et
            for komsu, sure in current.komsular:
                if komsu not in ziyaret_edildi:
                    # Hat değişimi varsa 5 dakika aktarma süresi ekle (Opsiyonel)
                    aktarma_suresi = 0 if current.hat == komsu.hat else 5
                    yeni_sure = toplam_sure + sure + aktarma_suresi
                    
                    yeni_rota = rota + [komsu]
                    heapq.heappush(pq, (yeni_sure, id(komsu), komsu, yeni_rota))
        
        # Yol bulunamadıysa None döndür
        return None

class MetroWithDLALT(MetroAgi):
    def __init__(self):
        super().__init__()
        self.landmarks = []
        self.dist_from_landmarks = {}
        self.dist_to_landmarks = {}
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_graph_data(self):
        """GNN için graf verisini hazırlar"""
        node_features = []
        node_mapping = {idx: i for i, idx in enumerate(self.istasyonlar.keys())}
        
        # Düğüm özellikleri: derece, hat bilgisi (one-hot encoded)
        for node in self.istasyonlar.values():
            # Basit özellikler: derece ve hat bilgisi
            degree = len(node.komsular)
            features = [degree]
            
            # Hat bilgisini one-hot encode etme (basit versiyon)
            hat_features = [0] * len(self.hatlar)
            if node.hat in self.hatlar:
                hat_idx = list(self.hatlar.keys()).index(node.hat)
                hat_features[hat_idx] = 1
            
            features.extend(hat_features)
            node_features.append(features)
        
        # Kenar listesi oluştur
        edge_index = []
        for i, (idx, node) in enumerate(self.istasyonlar.items()):
            for komsu, _ in node.komsular:
                j = node_mapping[komsu.idx]
                edge_index.append([i, j])
        
        # PyTorch Geometric Data objesi oluştur
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index).to(self.device)
    
    def train_landmark_selector(self, epochs=100):
        """Landmark seçici GNN'yi eğitir"""
        data = self.prepare_graph_data()
        
        # Modeli oluştur
        input_dim = data.x.size(1)
        self.gnn_model = GNNLandmarkSelector(input_dim, hidden_dim=16).to(self.device)
        optimizer = optim.Adam(self.gnn_model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Hedef: derecesi yüksek olan düğümler daha önemli
        target = torch.tensor(
            [min(len(list(self.istasyonlar.values())[i].komsular)/10, 1.0) 
            for i in range(data.x.size(0))], 
            dtype=torch.float
        ).to(self.device).unsqueeze(1)  # Boyut dönüşümü
        
        # Eğitim döngüsü
        self.gnn_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.gnn_model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def select_landmarks_with_gnn(self, k=4):
        """GNN ile en iyi landmark'ları seçer"""
        if not self.gnn_model:
            self.train_landmark_selector()
        
        data = self.prepare_graph_data()
        self.gnn_model.eval()
        with torch.no_grad():
            importance_scores = self.gnn_model(data).cpu().numpy().flatten()
        
        # Önem skoruna göre sırala ve en iyi k düğümü seç
        nodes = list(self.istasyonlar.values())
        sorted_indices = np.argsort(importance_scores)[::-1]  # Yüksekten düşüğe
        self.landmarks = [nodes[i] for i in sorted_indices[:k]]
        
        # Landmark'lar için mesafeleri hesapla
        for landmark in self.landmarks:
            self.dist_from_landmarks[landmark.idx] = self.dijkstra_from(landmark.idx)
            self.dist_to_landmarks[landmark.idx] = self.dijkstra_to(landmark.idx)
    
    def dijkstra_from(self, start_id: str) -> Dict[str, float]:
        """Bir düğümden tüm diğerlerine olan mesafeleri hesaplar"""
        distances = {start_id: 0}
        heap = [(0, start_id)]
        
        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist > distances[u]:
                continue
                
            for v, weight in self.istasyonlar[u].komsular:
                aktarma_suresi = 0 if self.istasyonlar[u].hat == v.hat else 5
                new_dist = current_dist + weight + aktarma_suresi
                if v.idx not in distances or new_dist < distances[v.idx]:
                    distances[v.idx] = new_dist
                    heapq.heappush(heap, (new_dist, v.idx))
        
        return distances
    
    def dijkstra_to(self, target_id: str) -> Dict[str, float]:
        """Tüm düğümlerden bir hedef düğüme olan mesafeleri hesaplar"""
        # Basit uygulama için normal Dijkstra kullanıyoruz
        # Gerçek uygulamada ters graf üzerinde çalıştırmak daha verimli olur
        return self.dijkstra_from(target_id)
    
    def dl_alt_heuristic(self, u: Istasyon, v: Istasyon) -> float:
        """GNN ile öğrenilmiş landmark'lara dayalı sezgisel fonksiyon"""
        max_h = 0
        for landmark in self.landmarks:
            # Üçgen eşitsizliği: d(u,v) ≥ |d(l,u) - d(l,v)|
            h = abs(self.dist_from_landmarks[landmark.idx].get(u.idx, float('inf')) - 
                   self.dist_from_landmarks[landmark.idx].get(v.idx, float('inf')))
            if h > max_h:
                max_h = h
        return max_h
    
    def dl_alt_search(self, start_id: str, target_id: str) -> Optional[Tuple[List[Istasyon], int]]:
        """DL-ALT ile en kısa yolu bulur"""
        if start_id not in self.istasyonlar or target_id not in self.istasyonlar:
            return None

        start = self.istasyonlar[start_id]
        target = self.istasyonlar[target_id]
        
        # Landmark'ları seç (henüz seçilmediyse)
        if not self.landmarks:
            self.select_landmarks_with_gnn()
        
        open_set = [(0 + self.dl_alt_heuristic(start, target), 0, id(start), start, [start])]
        heapq.heapify(open_set)
        
        g_scores = {start: 0}
        f_scores = {start: self.dl_alt_heuristic(start, target)}
        
        while open_set:
            _, g_score, _, current, rota = heapq.heappop(open_set)
            
            if current == target:
                return (rota, g_score)
            
            for neighbor, weight in current.komsular:
                aktarma_suresi = 0 if current.hat == neighbor.hat else 5
                tentative_g_score = g_score + weight + aktarma_suresi
                
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.dl_alt_heuristic(neighbor, target)
                    f_scores[neighbor] = f_score
                    heapq.heappush(open_set, (f_score, tentative_g_score, id(neighbor), neighbor, rota + [neighbor]))
        
        return None

# Örnek Kullanım
if __name__ == "__main__":
    print("\n=== DL-ALT Algoritması Testi ===")
    dl_metro = MetroWithDLALT()
    
    # Aynı istasyonları ve bağlantıları ekle
    # Kırmızı Hat
    dl_metro.istasyon_ekle("K1", "Kızılay", "Kırmızı Hat")
    dl_metro.istasyon_ekle("K2", "Ulus", "Kırmızı Hat")
    dl_metro.istasyon_ekle("K3", "Demetevler", "Kırmızı Hat")
    dl_metro.istasyon_ekle("K4", "OSB", "Kırmızı Hat")
    
    # Mavi Hat
    dl_metro.istasyon_ekle("M1", "AŞTİ", "Mavi Hat")
    dl_metro.istasyon_ekle("M2", "Kızılay", "Mavi Hat")
    dl_metro.istasyon_ekle("M3", "Sıhhiye", "Mavi Hat")
    dl_metro.istasyon_ekle("M4", "Gar", "Mavi Hat")
    
    # Turuncu Hat
    dl_metro.istasyon_ekle("T1", "Batıkent", "Turuncu Hat")
    dl_metro.istasyon_ekle("T2", "Demetevler", "Turuncu Hat")
    dl_metro.istasyon_ekle("T3", "Gar", "Turuncu Hat")
    dl_metro.istasyon_ekle("T4", "Keçiören", "Turuncu Hat")
    
    # Bağlantılar
    dl_metro.baglanti_ekle("K1", "K2", 4)
    dl_metro.baglanti_ekle("K2", "K3", 6)
    dl_metro.baglanti_ekle("K3", "K4", 8)
    dl_metro.baglanti_ekle("M1", "M2", 5)
    dl_metro.baglanti_ekle("M2", "M3", 3)
    dl_metro.baglanti_ekle("M3", "M4", 4)
    dl_metro.baglanti_ekle("T1", "T2", 7)
    dl_metro.baglanti_ekle("T2", "T3", 9)
    dl_metro.baglanti_ekle("T3", "T4", 5)
    dl_metro.baglanti_ekle("K1", "M2", 2)
    dl_metro.baglanti_ekle("K3", "T2", 3)
    dl_metro.baglanti_ekle("M4", "T3", 2)
    
    # Landmark seçiciyi eğit
    print("\nGNN Landmark Selector Eğitiliyor...")
    dl_metro.train_landmark_selector(epochs=50)
    
    # Test
    print("\nDL-ALT ile AŞTİ'den OSB'ye:")
    sonuc = dl_metro.dl_alt_search("M1", "K4")
    if sonuc:
        rota, sure = sonuc
        print(f"En hızlı rota ({sure} dakika):", " -> ".join(i.ad for i in rota))
    
    # Klasik Dijkstra ile karşılaştırma
    print("\nKlasik Dijkstra ile AŞTİ'den OSB'ye:")
    sonuc = dl_metro.en_hizli_rota_bul("M1", "K4")
    if sonuc:
        rota, sure = sonuc
        print(f"En hızlı rota ({sure} dakika):", " -> ".join(i.ad for i in rota))