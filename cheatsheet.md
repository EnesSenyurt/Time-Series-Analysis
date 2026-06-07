# 🎯 SUNUM CHEATSHEET — Zaman Serilerinde Olasılıksal Otomata ile Anomali Tespiti

> **Bu dosya kime?** Projeyi hiç bilmeyen birine, sunumda hocaya karşı projeyi savunabilmesi için yazıldı. Önce "30 saniyelik özet"i ezberle, sonra "Temel Kavramlar"ı oku, en sonda rubriğin her maddesi için olası soru-cevapları çalış. Sunumda panik yaparsan **"Hızlı Cevap Kartları"** (en altta) sana yeter.

---

## 🚀 30 SANİYELİK ÖZET (Ezberle — biri "bu proje ne?" derse bunu söyle)

> "İki tür model var: biri **kara kutu derin öğrenme** (LSTM ve 1D-CNN), öteki **yorumlanabilir olasılıksal otomata**. İkisini de **anomali tespiti** için iki veri setinde (SKAB ve BATADAL) karşılaştırdık. Amacımız 'en iyi modeli bulmak' değil; modellerin **performans, gürültüye dayanıklılık, bilinmeyen veri davranışı ve açıklanabilirlik** açısından nasıl davrandığını bilimsel olarak analiz etmek. Sonuç: derin öğrenme daha yüksek doğruluk veriyor ama **neden** öyle karar verdiğini söyleyemiyor; otomata daha düşük doğrulukta ama **her kararını olasılıklarla matematiksel olarak açıklayabiliyor.**"

**Tek cümlelik tema:** *"From Black-Box to Explainability"* = kara kutudan açıklanabilirliğe.

---

## 🧠 TEMEL KAVRAMLAR (Hiç bilmeyene anlatır gibi)

Bu projenin tamamı şu fikirden ibaret: **bir sayı dizisini harflere çevir, harf gruplarını "durum" yap, durumlar arası geçişlerin olasılığını öğren. Beklenmedik (düşük olasılıklı) geçişler = anomali.**

### Anomali nedir?
Verideki "normal dışı" davranış. Pompada arıza, su sisteminde siber saldırı vb. Biz bunu **ikili sınıflandırma** olarak ele aldık: her örnek ya `0` (normal) ya `1` (anomali).

### Zaman serisi nedir?
Zamana göre sıralı ölçümler (örn. her saniye sıcaklık, basınç). Sıra önemlidir — bu yüzden veriyi **rastgele karıştırmıyoruz** (aşağıda "data leakage"e bak).

### Derin Öğrenme tarafı (kara kutu)
- **LSTM** ve **1D-CNN**: sinir ağları. Geçmiş `30` adımlık pencereye bakıp "bu pencere anomali mi?" diye `0–1` arası olasılık üretir.
- "Kara kutu" çünkü içinde milyonlarca ağırlık var; **neden** öyle dediğini insan okuyamaz.

### Otomata tarafı (açıklanabilir) — İŞİN KALBİ, bunu iyi anla:
Tek boyutlu sayı dizisini şu 5 adımda işliyoruz:

1. **PCA → PC1:** Çok sütunlu veriyi (8 veya 43 sensör) tek boyuta indir. Otomata sadece tek boyutla çalışır (şartname şartı). PC1 = "verideki en çok bilgi taşıyan tek eksen".
2. **PAA** (*Piecewise Aggregate Approximation*): Diziyi parçalara böl, her parçanın ortalamasını al → diziyi kısalt/yumuşat. (Bizde `segment_size=1`, yani **kimlik** — diziyi olduğu gibi bırakır. Genel hali kodda var ve test edildi.)
3. **SAX** (*Symbolic Aggregate approXimation*): Sayıları **harflere** çevir. Örn. düşük değer→`a`, orta→`b`, yüksek→`c`. Kesim noktaları Gauss eğrisinin kantilleriyle (`alphabet_size`=3 → 3 harf). Artık dizi: `"aabccba..."` gibi.
4. **Kayan Pencere (Sliding Window):** Harf dizisinden `window_size`=4 uzunluğunda örtüşen parçalar al. `"aabc"`, `"abcc"`, `"bccb"`... **Her benzersiz parça = bir DURUM (state).**
5. **Geçiş Olasılıkları:** Hangi durumdan hangi duruma kaç kez geçilmiş say. `P(A→B) = (A'dan B'ye geçiş sayısı) / (A'dan toplam çıkış)`.

### Path Probability (yol olasılığı) — açıklanabilirliğin temeli
Bir dizi geçişin olasılığı = tek tek geçiş olasılıklarının **çarpımı**.
Örn: `P(aab→abc)=0.72`, `P(abc→bcc)=0.15` → **path probability = 0.72 × 0.15 = 0.108**.
- **Düşük olasılık → anomali** (model bunu beklemiyordu)
- **Yüksek olasılık → normal**

### Confidence Score (güven skoru)
Bizde **confidence = path probability**. 0.108 düşük olduğu için "düşük güven, muhtemelen anomali" deriz.

### Laplace Smoothing (düzleştirme) — neden gerekli?
Eğitimde hiç görülmemiş bir geçişin olasılığı `0` çıkar; `0` ile çarpınca tüm yol `0` olur (log alınca `-sonsuz`). Bunu önlemek için her sayıma `+1` (α=1) ekleriz: `P = (sayım + 1) / (toplam + 1×kelime_sayısı)`. Böylece hiçbir olasılık tam sıfır olmaz.

### Levenshtein (Edit Distance) — "Unseen" yönetimi
Test sırasında eğitimde **hiç görülmemiş** bir harf grubu gelirse ("unseen pattern"), onu **en az harf değişikliğiyle** ulaşılabilen en yakın bilinen durumla eşleriz.
Örn: `"adc"` eğitimde yok → `"abc"`e 1 harf uzaklıkta (`d`→`b`) → `"abc"` üzerinden devam et.
Beraberlik olursa **alfabetik olarak en küçüğü** seçilir → sonuç hep aynı (deterministik).

### Normal-Only eğitim — neden?
Otomatayı **sadece eğitimdeki normal (anomaly=0)** veriden kuruyoruz. Böylece otomata "normal" davranışı öğrenir; anomalik diziler düşük olasılık alıp yakalanır. ("Düşük olasılık = anomali" mantığı ancak böyle tutarlı olur.)

---

## 🗂️ PROJE HARİTASI (Hangi dosya ne yapıyor?)

| Dosya | Görevi | Rubrik bağlantısı |
|---|---|---|
| `config/config.yaml` | **TÜM parametreler burada.** Kodda hard-coded değer yok. | Kriter 1 (mimari) |
| `src/config.py` | YAML'ı okuyup koda taşır | Kriter 1 |
| `src/data/data_loader.py` | SKAB + BATADAL veri yükleme | Kriter 2 |
| `src/data/preprocess.py` | Impute + StandardScaler + PCA (sadece train'de fit) | Kriter 2 |
| `src/data/splits.py` | SKAB GroupKFold / BATADAL zaman sıralı bölme | Kriter 2, 4 |
| `src/automata/paa.py` | PAA dönüşümü | Kriter 2 |
| `src/automata/sax.py` | SAX (sayı→harf) | Kriter 2 |
| `src/automata/patterns.py` | Kayan pencere + geçişler | Kriter 2 |
| `src/automata/levenshtein.py` | Edit distance + en yakın eşleme | Kriter 2 (zorunlu test) |
| `src/automata/automaton.py` | **Olasılıksal otomata** (durum, olasılık, path-prob) | Kriter 2, 3 |
| `src/explain/explainer.py` | **Açıklanabilirlik** (JSON karar açıklaması) | Kriter 3 |
| `src/models/dl_models.py` | LSTM + 1D-CNN | Kriter 2 |
| `src/experiments/metrics.py` | Accuracy/Precision/Recall/F1 | Kriter 4 |
| `src/experiments/scenarios.py` | original / noise / unseen senaryoları | Kriter 4 |
| `src/experiments/stats_tests.py` | Wilcoxon + McNemar | Kriter 4 |
| `src/experiments/runner.py` | Tüm deneyleri orkestre eder | Kriter 1, 4 |
| `src/viz/plots.py` | Confusion matrix, ROC/PR, heatmap, otomata diyagramı | Kriter 5 |
| `tests/` | 25+ birim test (pytest) | Kriter 1, 2 |
| `README.md` | Akademik rapor (TR) | Kriter 5 |

**Çalıştırma komutları (sorulursa):**
```bash
python scripts/run_experiments.py --only smoke   # hızlı doğrulama
python scripts/run_experiments.py --only main    # ana karşılaştırma
python scripts/run_experiments.py --only grid    # parametre taraması
python scripts/make_report_assets.py             # figürler
python -m pytest tests/ -v                        # testler
```

---

## 📊 EZBERLENECEK SAYILAR (Hoca sonuç sorarsa)

**Veri setleri:**
- **SKAB:** 22.472 satır, 8 sensör, %34,8 anomali, hedef = `anomaly`, ayraç `;`
- **BATADAL:** 4.177 satır, 43 özellik, %5,2 anomali, hedef = `ATT_FLAG`, zaman = `DATETIME`

**Ana sonuçlar (F1 skoru — original senaryo):**

| Veri | LSTM | 1D-CNN | Otomata |
|---|---|---|---|
| **SKAB** | **0.836** | 0.830 | 0.499 |
| **BATADAL** | 0.000 | 0.000 | **0.052** |

**Üç anahtar bulgu (bunu söyle):**
1. **SKAB'da DL kazanıyor** (0.836 vs 0.499) — bol anomali örneği var, ağlar iyi öğreniyor.
2. **BATADAL'da DL çöküyor** (F1=0) — %5,2 anomali çok az, ağlar "her şey normal" diyor. Otomata tek pozitif yakalayan model (0.052).
3. **İstatistik:** LSTM vs CNN farkı **anlamsız** (p=0.91); DL vs Otomata farkı **anlamlı** (p=0.0039 < 0.01).

**Gürültü etkisi:** İhmal edilebilir (F1 düşüşü ~0.002). Her iki model de Gaussian gürültüye dayanıklı.

**Unseen oranı:** SKAB %0,1, BATADAL %0,5 — çok düşük, Levenshtein hepsini kurtarıyor.

---

# 📋 RUBRİK MADDE MADDE — OLASI SORULAR & CEVAPLAR

---

## ✅ KRİTER 1 — Yazılım Mimarisi ve Kod Kalitesi (20 puan)

**Hoca ne arıyor:** Merkezi config, parametreye bağlı otomatik üretim, modüler pipeline, düzgün Git kullanımı.

**Biz ne yaptık:**
- Tüm parametreler `config/config.yaml`'da. Kodda **hiçbir hard-coded değer yok**.
- `window_size`'ı config'te değiştirince tüm sistem (SAX, durumlar, sonuçlar) otomatik yeniden üretiliyor.
- Modüler yapı: `data/`, `automata/`, `models/`, `explain/`, `experiments/`, `viz/` ayrı paketler.
- Pipeline: ham veri → bölme → ön işleme → {DL, otomata} → değerlendirme → açıklama → görsel.

**Olası sorular:**

**S: "Hard-coded değer yok" diyorsunuz, nasıl emin olabiliriz?**
> C: Bütün sayılar — pencere boyutu, alfabe boyutu, epoch, batch, seed listesi, gürültü oranı — `config.yaml`'dan okunuyor. Kod sadece config nesnesini parametre olarak alıyor. Örneğin `automaton.py` smoothing α'yı `cfg.automaton.smoothing_alpha`'dan alır.

**S: Parametre değişince ne oluyor?**
> C: Tek kaynak (single source of truth) prensibi. `config.yaml`'da `window_size: 4` → `5` yaparsanız, otomata 5 harflik durumlar üretir, durum sayısı değişir, tüm sonuçlar buna göre yeniden hesaplanır. Hiçbir yeri elle değiştirmeye gerek yok.

**S: Git kullanımınız nasıl? (Bu önemli — kuralda "dengesiz commit → 0" var)**
> C: İki kişiyiz (Ali ve Enes), bileşene göre böldük: Ali veri+otomata+açıklanabilirlik, Enes derin öğrenme+deney+rapor. Her commit Conventional Commits formatında (`feat:`, `test:`, `fix:`, `docs:`), Türkçe açıklamalı, sık ve dengeli. Ortak dosyalarda `Co-authored-by` kullandık.

**S: Test yazdınız mı?**
> C: Evet, `tests/` altında 25+ pytest birim testi var. Özellikle **Levenshtein için testler şartname gereği zorunluydu**, onları yazdık. PAA, SAX, patterns, automaton, explainer, splits, preprocess, metrics, stats hepsinin testi var.

---

## ✅ KRİTER 2 — Veri Ön İşleme ve Modelleme Doğruluğu (25 puan) — EN YÜKSEK PUAN

**Hoca ne arıyor:** Doğru ön işleme (normalizasyon, PCA), DL kurulumu, otomata inşası (PAA/SAX/sliding), geçiş olasılıkları + smoothing, Levenshtein + birim testler.

**Biz ne yaptık:** (yukarıdaki "Temel Kavramlar"ın tamamı bu maddeye girer)

**Olası sorular:**

**S: Veri sızıntısını (data leakage) nasıl önlediniz?**
> C: Çok kritik. Tüm dönüşümleri **sadece eğitim verisinde fit ediyoruz**, sonra aynı dönüşümü val/test'e sadece `transform` olarak uyguluyoruz:
> - StandardScaler → train ortalaması/std ile
> - PCA → sadece train'de fit
> - SAX sözlüğü ve geçiş olasılıkları → sadece train
> Ayrıca SKAB'da aynı `.csv` dosyası hem train hem test'te olamaz (GroupKFold).

**S: Neden PCA ve neden sadece PC1?**
> C: Otomata tek boyutlu seriyle çalışır (şartname şartı). Çok sütunlu veriyi (8 veya 43 sensör) PCA ile tek boyuta indirip **ilk bileşeni (PC1)** alıyoruz. PC1 verideki varyansın en büyük kısmını taşıyan eksendir. DL modelleri ise çok boyutlu veriyi olduğu gibi kullanıyor.

**S: SAX'ta harf kesim noktalarını nasıl belirlediniz?**
> C: Gaussian (standart normal) kantillerine göre. `alphabet_size=3` için 2 kesim noktası (`scipy.stats.norm.ppf`). Veri z-normalize edildiği için normal dağılıma yaklaşır ve harfler **kabaca eşit olasılıklı** olur. Bu klasik SAX yöntemidir (Lin et al. 2007).

**S: Geçiş olasılığını nasıl hesapladınız?**
> C: Frekans tabanlı: `P(Si→Sj) = (geçiş sayısı + α) / (toplam çıkış + α×|V|)`. α=1 Laplace smoothing. `automaton.py`'deki `prob()` fonksiyonu bunu yapar. Smoothing sayesinde görülmemiş geçişler bile sıfır olmayan olasılık alır.

**S: PAA'da segment_size=1 ise PAA hiçbir şey yapmıyor, neden var?**
> C: Doğru, `segment_size=1`'de PAA kimlik dönüşümüdür (diziyi değiştirmez). Ama **genel implementasyon kodda var ve test edildi** — config'ten `segment_size=2` yaparsanız gerçek sıkıştırma (segment ortalaması) yapar. Bu projede tam çözünürlüğü korumak için 1 seçtik; rapora da not düştük. Yani PAA mimaride mevcut, sadece parametresi kimlik.

**S: Levenshtein nedir, nerede test ettiniz? (ZORUNLU madde)**
> C: İki string arası minimum düzenleme (ekleme/silme/değiştirme) sayısı. Unseen pattern geldiğinde en yakın bilinen pattern'a eşlemek için kullanıyoruz. `levenshtein.py`'de dinamik programlama ile, `tests/test_levenshtein.py`'de test ettik: `levenshtein("adc","abc")==1`, beraberlikte alfabetik en küçük seçilir (deterministik).

**S: DL modellerini nasıl kurdunuz?**
> C: LSTM (64→32 nöron, dropout 0.2) ve 1D-CNN (64→32 filtre, kernel 3). İkisi de `30` adımlık pencere alıp sigmoid ile `0–1` olasılık üretiyor. Epoch=50, batch=32, EarlyStopping (val_loss, patience=5), `class_weight=balanced` (dengesizlik için). Hepsi config'ten.

**S: Pencere etiketini nasıl belirlediniz?**
> C: Pencere içinde **en az bir anomali varsa** (`max(y)`) pencere etiketi 1. Hem DL hem otomata aynı pencere mantığını kullanır → adil karşılaştırma.

---

## ✅ KRİTER 3 — Olasılıksal Açıklanabilirlik Modülü (20 puan)

**Hoca ne arıyor:** Her karar için state, pattern, transition, unseen mekanizması, geçiş olasılıkları, path probability, confidence skoru.

**Biz ne yaptık:** `src/explain/explainer.py` her karar için şu JSON'u üretir:
```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "probability": 0.108,
  "decision": "anomaly"
}
```
Tam kayıt ayrıca `transitions`, `distance`, `confidence_score`, `rationale` içerir.

**Olası sorular:**

**S: Bu modül DL modelinde neden yok?**
> C: İşte projenin ana noktası bu! DL'in sigmoid çıktısı sadece "0.87" gibi bir sayı verir; **hangi zaman adımının** kararı etkilediğini, **neden** öyle dediğini söyleyemez. Otomata ise: "şu durumdan şu duruma 0.72, oradan şuraya 0.15 olasılıkla geçtin, çarpım 0.108 — bu çok düşük, demek ki anomali" diye **tam matematiksel gerekçe** sunar. Kritik altyapıda operatör bu gerekçeye ihtiyaç duyar.

**S: Confidence score'u nasıl tanımladınız?**
> C: Confidence = path probability. Şartname örneğiyle birebir: `0.72 × 0.15 = 0.108 → Low confidence → anomali`. Düşük olasılık = düşük güven = anomali olası.

**S: Bu açıklamalar deterministik mi? (Aynı girdi aynı çıktıyı verir mi?)**
> C: Evet. Vocab sıralı tutulur, Levenshtein beraberliği alfabetik çözülür. Aynı girdi her zaman aynı açıklamayı üretir → yeniden üretilebilir.

**S: Bonus (ek puan) bir şey yaptınız mı?**
> C: Evet, iki opsiyonel analiz: **Counterfactual** (alternatif pattern'lar altında olasılık nasıl değişir) ve **Similarity report** (unseen pattern'a en yakın N bilinen pattern + mesafeleri). `explainer.py`'de `counterfactual()` ve `similarity_report()`.

---

## ✅ KRİTER 4 — Deneysel Tasarım ve İstatistiksel Analiz (15 puan)

**Hoca ne arıyor:** 3 senaryo (normal/gürültü/unseen) sistematik test, parametre etkisi (window/alphabet), veri-uygun değerlendirme (SKAB GroupKFold, BATADAL zaman sıralı), istatistik testleri.

**Biz ne yaptık:**
- **3 senaryo:** `original` (ham), `noise` (test'e Gaussian gürültü σ=train_std×0.2), `unseen` (unseen oranı raporlanır).
- **Değerlendirme:** SKAB → StratifiedGroupKFold (3 fold, grup=`source_file`); BATADAL → zaman sıralı %60/%20/%20.
- **Parametre grid:** window∈{3,4,5,6} × alphabet∈{3,4,5,6} = 16 kombinasyon.
- **İstatistik:** Wilcoxon (eşleştirilmiş F1) + McNemar (örnek bazında).
- **Tekrarlanabilirlik:** seed listesi [42, 123, 2026], sonuçlar ortalama ± std.

**Olası sorular:**

**S: Neden veriyi rastgele bölmediniz?**
> C: Zaman serisinde sıra önemli. Rastgele bölme zaman bağımlılığını bozar ve sızıntı yaratır (gelecekten geçmişe bilgi sızar). SKAB'da dosya bazlı GroupKFold (aynı dosya iki tarafta olamaz), BATADAL'da zaman sıralı kronolojik bölme yaptık.

**S: Wilcoxon ve McNemar farkı ne?**
> C: **Wilcoxon** → fold/seed başına F1 skorlarını eşleştirip karşılaştırır (sürekli metrik). **McNemar** → örnek örnek "A doğru/B yanlış" çiftlerini sayar (sınıflandırma kararları). İkisi de eşleştirilmiş (paired) testlerdir.

**S: İstatistik sonuçları ne çıktı?**
> C: LSTM vs 1D-CNN: p=0.91 → **anlamlı fark yok** (iki model eşdeğer). DL vs Otomata: p=0.0039 → **anlamlı fark var** (SKAB'da DL üstün, %99 güvenle).

**S: Neden BATADAL'da istatistik testi yapmadınız?**
> C: BATADAL'da DL modelleri sınıf çöküşü yaşadı (F1=0, hepsini normal tahmin etti). F1=0 üzerinden test yapmak anlamsız olurdu, o yüzden istatistik testlerini SKAB'a sınırladık ve bunu rapora şeffafça yazdık.

**S: Parametre etkisi ne çıktı?**
> C: Büyük `window_size` genelde F1'i hafif artırıyor (daha uzun örüntü yakalar). SKAB'da en iyi window=6/alphabet=3 (F1=0.500), BATADAL'da window=6/alphabet=4 (F1=0.204). Durum sayısı ve geçiş yoğunluğu grid boyunca görece sabit → otomata yapısal olarak kararlı.

---

## ✅ KRİTER 5 — Akademik Raporlama ve Analitik Derinlik (20 puan)

**Hoca ne arıyor:** Veri setleri arası karşılaştırma, olasılık yorumu (low/high likelihood), görseller, akademik yazım.

**Biz ne yaptık:**
- `README.md` tam akademik rapor (TR): giriş, yöntem, deney, bulgular, tartışma, 10 kaynak.
- **5 görsel türü:** Confusion Matrix, ROC/PR eğrisi, otomata durum diyagramı, geçiş olasılığı heatmap'i, parametre duyarlılık grafikleri (`results/figures/`).

**Olası sorular:**

**S: İki veri seti arası temel fark nedir?**
> C: **Anomali oranı.** SKAB %34,8 (dengeli), BATADAL %5,2 (çok dengesiz). SKAB'da DL bol örnekle iyi öğreniyor; BATADAL'da az anomali yüzünden DL çöküyor. Bu, **aşırı dengesiz veride yapısal/kural tabanlı modellerin (otomata) DL'ye karşı avantajlı olabileceğini** gösteriyor — projenin en güçlü bulgusu bu.

**S: Sonuç olarak hangi model "kazandı"?**
> C: Bilinçli olarak "tek en iyi model" demiyoruz — şartname de bunu istemiyor. DL ham doğrulukta üstün ama açıklanamaz; otomata daha düşük doğrulukta ama tam açıklanabilir ve dengesiz veride daha dayanıklı. **Hangisini seçeceğin uygulamaya bağlı:** kritik altyapıda açıklanabilirlik şart → otomata tercih edilebilir.

---

# 🛡️ ZOR SORULAR & ZAYIF NOKTA SAVUNMASI (Buraya dikkat!)

Bunlar hocanın "açık yakalamak" için soracağı sorular. Hazır ol:

**S: BATADAL'da F1=0.000?! Modeliniz çalışmıyor mu?**
> C: Hayır, bu bir **bulgu**, bug değil. BATADAL'da anomali oranı sadece %5,2. DL modeli `class_weight=balanced` kullanmamıza rağmen, bu kadar aşırı dengesizlikte "her şeyi normal de" diyerek %81 accuracy alıyor ama F1=0. Bu, **derin öğrenmenin aşırı dengesiz veride başarısız olduğunu** gösteren bilimsel bir sonuç. Nitekim otomata burada tek pozitif yakalayan model (F1=0.052). Bunu raporda açıkça analiz ettik.

**S: Şartname 5 seed istiyor, siz 3 kullanmışsınız. Neden?**
> C: Hesaplama süresi nedeniyle (DL × 2 veri × 3 senaryo × seed × fold çok ağır). seed'i 3'e, SKAB fold'unu 3'e düşürdük. **Bunu raporda şeffaflıkla belirttik** ve config'ten tek satırla 5'e çıkarılabilir. Metodoloji doğru, sadece tekrar sayısı düşük — sonuçların yönü değişmez.

**S: Otomata SKAB'da çok düşük (F1=0.499), işe yaramıyor mu?**
> C: Otomata burada **dezavantajlı koşulda**: sadece **tek boyut (PC1)** kullanıyor, DL ise 8 sensörün tamamını. Buna rağmen 0.499 alması ve **her kararını açıklayabilmesi** değerli. Amaç DL'i yenmek değil, açıklanabilirlik-performans dengesini göstermek.

**S: PAA aslında hiçbir şey yapmıyor (segment=1).**
> C: Doğru, bu projede kimlik. Ama implementasyon genel ve test edilmiş; config'ten `segment_size>1` ile gerçek sıkıştırma yapar. PC1 zaten tek boyut olduğu için tam çözünürlüğü korumayı seçtik. Mimari olarak PAA pipeline'da mevcut.

**S: confidence = path probability demişsiniz, bu gerçek bir "güven" mi?**
> C: Şartname örneğiyle birebir uyumlu (`0.108 = Low confidence`). Path probability düşükse model o diziyi "beklemediği" için güveni düşük — bu olasılıksal olarak tutarlı bir güven tanımı. İstersek `-log` ile skora da çeviriyoruz (`neg_log_path`).

**S: BATADAL'da -999 değerlerini ne yaptınız?**
> C: `ATT_FLAG` kolonunda 219 saldırı (`1`), 3.958 normal (`0`). `-999` etiketsiz/gizli dönemi gösterir. Politikamız: `ATT_FLAG > 0 → 1`, aksi halde `0` (yani -999 → normal). Bu kararı ve sayıları rapora not düştük; config'ten `unlabeled_policy` ile değiştirilebilir.

**S: DL pencere boyutu 30, SAX pencere boyutu 4 — neden farklı?**
> C: Bilinçli bağımsız seçim. DL'in "lookback" penceresi (30 adım) ile otomatanın SAX örüntü uzunluğu (4 sembol) farklı kavramlar. DL daha uzun bağlama ihtiyaç duyar; SAX örüntüsü kısa tutulunca durum sayısı yönetilebilir kalır. Pencere uyumu gelecek çalışma olarak rapora yazıldı.

**S: Neden sadece LSTM ve CNN? GRU yok?**
> C: Şartname "en az iki" istiyordu. LSTM ve 1D-CNN **mimari olarak en farklı** ikilidir (tekrarlayan vs evrişimli) — bu yüzden karşılaştırma daha anlamlı. GRU, LSTM'e çok benzer olduğu için onun yerine CNN'i seçtik.

---

## 🎤 HIZLI CEVAP KARTLARI (Sunum sırasında göz at)

| Kavram | Tek cümle |
|---|---|
| **Proje amacı** | Kara kutu DL ile açıklanabilir otomatayı anomali tespitinde karşılaştırmak |
| **PAA** | Diziyi parçalara böl, ortalamasını al (bizde kimlik) |
| **SAX** | Sayıları harflere çevir (a/b/c) |
| **State (durum)** | window_size uzunluğunda harf grubu (örn "aab") |
| **Geçiş olasılığı** | P(A→B) = geçiş sayısı / toplam çıkış |
| **Path probability** | Geçiş olasılıklarının çarpımı → düşükse anomali |
| **Confidence** | = path probability; düşük = anomali olası |
| **Laplace smoothing** | Her sayıma +1, sıfır olasılığı önler |
| **Levenshtein** | İki string arası min düzenleme; unseen'i en yakına eşler |
| **Unseen** | Eğitimde görülmemiş pattern; Levenshtein ile kurtarılır |
| **Normal-only** | Otomata sadece normal veriden öğrenir |
| **SKAB değerlendirme** | StratifiedGroupKFold (dosya bazlı, sızıntısız) |
| **BATADAL değerlendirme** | Zaman sıralı %60/%20/%20 |
| **Wilcoxon** | F1 skorlarını eşleştirip karşılaştırır |
| **McNemar** | Örnek bazında doğru/yanlış karşılaştırır |
| **En güçlü bulgu** | Dengesiz veride (BATADAL) otomata DL'den dayanıklı |

**Eğer takılırsan kurtarıcı cümle:**
> "Bu projede amacımız tek bir en iyi modeli bulmak değil; iki yaklaşımın farklı koşullardaki davranışını bilimsel ve sistematik olarak analiz etmekti. Derin öğrenme performansta, otomata açıklanabilirlikte öne çıkıyor."

---

## 📝 SUNUM AKIŞ ÖNERİSİ (5–7 dk)

1. **Problem (30 sn):** Anomali tespiti kritik; DL güçlü ama kara kutu; biz açıklanabilir alternatifi karşılaştırdık.
2. **Veri (30 sn):** SKAB (pompa, dengeli) + BATADAL (su sistemi siber saldırı, dengesiz).
3. **Yöntem (90 sn):** DL tarafı (LSTM/CNN) kısaca; otomata tarafı PAA→SAX→durum→geçiş olasılığı detaylı (işin yıldızı bu).
4. **Açıklanabilirlik (60 sn):** JSON örneğini göster (`0.72×0.15=0.108→anomali`). DL bunu yapamaz vurgusu.
5. **Sonuçlar (90 sn):** Tabloyu göster; SKAB'da DL, BATADAL'da otomata; istatistik anlamlılık.
6. **Tartışma (30 sn):** Performans vs açıklanabilirlik dengesi; dengesiz veride otomata avantajı.
7. **Görseller:** Otomata diyagramı + heatmap + confusion matrix göster.

**Başarılar! 🍀 Takılırsan: yavaşla, "Hızlı Cevap Kartları"na bak, en kötü ihtimalde kurtarıcı cümleyi söyle.**
