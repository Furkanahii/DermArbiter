# DermAbench v1-lite — Klinisyen Anotasyon Föyü (B3)

Merhaba Abdurrahim Hocam,

Bu paket, DermAbench benchmark'ının ilk küçük (v1-lite) sürümü için
**gold-standard** etiketlemenizi rica ettiğimiz vakaları içeriyor.

## Paket içeriği

- `dermabench_v1lite_worksheet.csv` — doldurmanızı rica ettiğimiz föy (198 vaka)
- `dermabench_v1lite.jsonl` — (teknik) ham vaka verisi, sizin doldurmanıza gerek yok

## Vaka seçimi

198 vaka, **SCIN** (Google'ın halka açık dermatoloji veri seti) içinden
seçildi. Fitzpatrick deri tipine göre **dengeli** örnekledik — koyu deri
(Tip V–VI) gerçek veride çok az olduğu için, adillik (fairness) ölçümünde
anlamlı bir N kalsın diye o grupları olabildiğince tam aldık:

| Fitzpatrick | I | II | III | IV | V | VI |
|---|---|---|---|---|---|---|
| Vaka | 33 | 39 | 33 | 33 | 33 | 27 |

117 farklı klinik tanı temsil ediliyor (egzama, ürtiker, herpes vb.).

## CSV nasıl doldurulur

Excel veya Google Sheets ile açın. Her satır bir vaka:

- **image_url** — hücredeki bağlantıya tıklayın; lezyon görüntüsü tarayıcıda
  açılır (indirme gerekmez, halka açık).
- **clinical_history** — hastanın bildirdiği semptom/süre bilgisi.
- **auto_diagnosis / auto_icd10 / auto_management** — SCIN'in ham etiketi ve
  otomatik ürettiğimiz ön-öneriler. **Sadece referans**; lütfen kendi
  değerlendirmenizi yazın.

Doldurmanızı rica ettiğimiz sütunlar:

| Sütun | Ne yazılmalı |
|---|---|
| `ref_dx_1` | Birincil (en olası) tanınız — *zorunlu* |
| `ref_dx_2`, `ref_dx_3` | Ayırıcı tanıda 2. ve 3. olasılıklar (varsa) |
| `management` | `biopsy` / `monitor` / `reassure` (üçünden biri) |
| `is_malignant` | `Y` (malign) / `N` (benign) |
| `approve` | `Y` = vaka kullanılabilir / `N` = görüntü kötü, hariç tut |
| `notes` | Serbest not (opsiyonel) |

> Görüntü kalitesi yetersizse veya tanı görüntüden verilemiyorsa
> `approve = N` işaretlemeniz yeterli — o vakayı setten çıkarırız.

## Doldurduktan sonra

CSV'yi bize geri gönderin; biz şu komutla dondurulmuş (frozen) gold
setine çeviriyoruz:

```
python scripts/curate_dermabench.py apply \
  --subset data/dermabench/dermabench_v1lite.jsonl \
  --worksheet <sizin_doldurdugunuz>.csv \
  --out data/dermabench/dermabench_v1lite_frozen.jsonl
```

Yalnızca `approve = Y` olan vakalar frozen sete girer; sizin yazdığınız
ayırıcı tanı, management ve malignite alanları skorlamanın
referans-doğrusu (ground truth) olur.

Teşekkürler,
Emre & Furkan
