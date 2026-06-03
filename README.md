# Time-Series-Analysis

Bu repo, YazLab zaman serisi anomali tespiti projesi icin BATADAL ve SKAB veri setlerini birlikte kullanacak sekilde duzenlenmistir.

## Veri setleri

### BATADAL

`BATADAL/` klasoru proje icinde hazir duruyor.

- `BATADAL_dataset03.csv`
- `BATADAL_dataset04.csv`
- `Attacks_TrainingDataset2.jpg`

BATADAL CSV dosyalari virgul ile ayrilmis ve etiket kolonu `ATT_FLAG` olarak geliyor.

### SKAB

`SKAB/` klasoru, `waico/SKAB` reposundaki `data/` klasorunden alinmistir:

https://github.com/waico/SKAB/tree/master/data

Klasor yapisi:

- `SKAB/anomaly-free/anomaly-free.csv`: normal calisma verisi
- `SKAB/valve1/*.csv`: pompa giris valfi deneyleri
- `SKAB/valve2/*.csv`: pompa cikis valfi deneyleri
- `SKAB/other/*.csv`: sizinti, rotor dengesizligi, kavitation vb. diger deneyler

SKAB CSV dosyalari noktalı virgul (`;`) ile ayrilmis. Arizali deney dosyalarinda `anomaly` ve `changepoint` etiketleri var. `anomaly-free` dosyasinda etiket kolonlari yok; loader bu dosyaya otomatik olarak `anomaly = 0` ve `changepoint = 0` ekler.

## Hızlı kullanım

Gerekirse pandas kur:

```powershell
python -m pip install pandas
```

Verileri okumak icin:

```python
from src.data_loader import load_batadal_all, load_skab_all

batadal = load_batadal_all()
skab = load_skab_all()

print(batadal.shape)
print(skab.shape)
```

## Sonraki geliştirme adımları

1. BATADAL icin `ATT_FLAG`, SKAB icin `anomaly` kolonunu ortak hedef etikete donustur.
2. Zaman kolonlarini normalize et: BATADAL `DATETIME`, SKAB `datetime`.
3. Her veri seti icin ayri EDA notebook'u hazirla.
4. Baseline modellerle basla: Isolation Forest, One-Class SVM, PCA reconstruction error.
5. Sonra LSTM Autoencoder veya Conv-AE gibi derin ogrenme modellerine gec.
