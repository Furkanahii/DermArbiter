# DermAbench — Ölçekleme Planı (v1-lite → v1 → v2)

> Amaç: v1-lite (240 vaka, pipeline doğrulama) üzerinden tam ölçekli,
> istatistiksel olarak güçlü, fair ve yayına hazır bir benchmark'a geçiş.
> Felsefe: **önce küçük & doğru, sonra büyüt** — her aşama bir öncekini
> doğrulayıp ölçeği artırır.

---

## 0. Mevcut durum (envanter)

| Havuz | Vaka | Fitzpatrick dağılımı | Malign | Görüntü (lokal) |
|---|---|---|---|---|
| SCIN | 3.061 | I=334, II=1007, III=859, IV=435, V=229, **VI=27**, ?=170 | 7 | 165 |
| DDI  | 656   | II=208, III=241, V=207 | 171 | 656 (tümü) |
| **v1-lite (curated)** | **240** | I=40,II=53,III=40,IV=40,V=40,VI=27 | 60 | 240 (tümü) |

**Birleşik erişilebilir N (SCIN+DDI):** I≈334, II≈1215, III≈1100, IV≈435,
V≈436, **VI=27**, malign≈178.

---

## 1. Bağlayıcı kısıtlar (ölçeği bunlar belirliyor)

1. **Fitzpatrick VI darboğazı** — tüm havuzda yalnızca **27** VI vakası var.
   Bu, fairness'ın en kritik (en koyu, en az hizmet alan) grubu. Mevcut
   veriyle aşılamaz → yeni koyu-deri kaynağı gerekir (bkz. §5).
2. **Malign tavanı** — biyopsi-doğrulamalı malign en fazla **178** (171 DDI
   + 7 SCIN). Safety boyutunun üst sınırı budur.
3. **Compute** — agentic koşum vaka başına ~2-4 dk (A100, 4 ajan × 5 faz).
   240→~9-17h; 1.000→~33-67h. Batch + `--resume` zorunlu.
4. **Klinisyen kapasitesi** — bir dermatolog 1.000+ vakayı elle dondu-
   ramaz. Katmanlı anotasyon gerekir (bkz. §4).

---

## 2. Hedef ölçekler ve istatistiksel gerekçe

Oran tahmininde 95% CI yarı-genişliği: `n ≈ 0.96 / E²` (en kötü p=0.5).

| Aşama | N | Grup başı (Fitz) | Alt-grup CI | Malign | Durum |
|---|---|---|---|---|---|
| **v1-lite** | 240 | ~40 | ±15pp | 60 | ✅ pipeline doğrulama |
| **v1** | **~1.000** | ~150 (VI=27) | ±8pp (VI: ±19pp*) | ~150 | 🎯 ana sonuçlar |
| **v2** | ~2.000+ | ~250 (+yeni VI) | ±6pp | ~250 | makale, frozen gold |

\* VI=27 ile CI geniş kalır → v1'de **keşifsel (exploratory)** olarak
raporlanır; V+VI "dark" altında birleştirilerek primary fairness ölçülür.

**Neden ~1.000 (v1):** grup başına ~150 vaka → alt-grup doğruluğunu ±8pp
ile tahmin edebiliriz; light-vs-dark açığı için yeterli güç. ~150 malign →
triyaj duyarlılığı (~0.9) ±5pp. VI hâlâ 27'de tıkalı (v2'ye taşınır).

---

## 3. Kaynak tahsisi (v1, ~1.000 vaka)

Fairness-aware curation (eşit kota + nadir grubu doldur) ile:

| Fitzpatrick | Hedef | Kaynak | Not |
|---|---|---|---|
| I   | 150 | SCIN | |
| II  | 150 | SCIN + DDI | |
| III | 150 | SCIN + DDI | |
| IV  | 150 | SCIN | |
| V   | 150 | SCIN + DDI | DDI malign ağırlıklı |
| VI  | 27  | SCIN | **tümü** (darboğaz) |
| redistribute | ~220 | büyük gruplara | slack VI'dan dağıtılır |

- `--min-malignant 150` → safety için taban (Fitz dengesi korunarak).
- `--target-n 1000 --seed 42` → tekrarlanabilir.
- İsteğe bağlı: **Derm1M** havuzu eklenirse (loader hazır) hem hacim hem
  kondisyon çeşitliliği artar.

---

## 4. Anotasyon katmanları (B3 ölçeklemesi)

1.000 vakanın hepsini klinisyen dondurmaz. Katmanlı strateji:

- **Tüm 1.000 → silver** (dataset etiketleri: SCIN konsensüs + DDI patoloji).
  Şimdi skorlanabilir.
- **Öncelikli alt-küme → clinician-frozen** (~300-400 vaka): tüm malign +
  tüm V/VI koyu deri + ajanların çeliştiği belirsiz vakalar. Dermatolog
  bunu gözden geçirebilir.
- Skorlamada her vaka kendi statüsüyle raporlanır; "frozen-only" ve
  "silver-dahil" sonuçlar ayrı sütun olarak verilir.

---

## 5. Fitzpatrick VI / koyu-deri genişletme (v2 için)

VI darboğazını aşmak için ek kaynaklar (öncelik sırası):

1. **DDI-2** (Stanford, Asyalı hastalar) — gated, aynı portal.
2. **Fitzpatrick17k** — repoda loader paterni var; çok sayıda koyu-deri.
3. **PubMed dermatoloji** — `load_pubmed` stub'ı implemente edilecek.
4. Hedef: VI'yı ≥100'e çıkarmak → fairness'ı VI için de güç kazandırmak.

---

## 6. Compute / maliyet bütçesi

- **Yerel modeller** (MedGemma-4B, Qwen3-8B 4-bit): A100'de ücretsiz,
  ama zaman maliyeti. 1.000 vaka ≈ 33-67h → 3-5 oturuma böl, `--resume`.
- **Gemini API** (Specialist + Moderator): vaka başına ~2 çağrı. Bütçe
  Abdurrahim'den. GPT-4o baseline'ı için ayrı key.
- **Checkpoint**: `--dermabench-out` zaten vaka başına fsync yazıyor →
  koşum yarıda kesilse de ilerleme korunur.
- **Görüntü indirme**: v1 SCIN alt-kümesi (~830 görüntü, ~700MB) curation
  sonrası indirilir (DDI tümü zaten lokal).

---

## 7. Yürütme adımları (v1)

```bash
# 1) (ops.) Derm1M havuzunu da üret — daha fazla hacim/çeşitlilik
python scripts/build_dermabench.py --source derm1m \
    --raw-dir data/dermabench/raw/derm1m --out data/dermabench/derm1m.jsonl

# 2) v1 setini curate et (fairness + malign tabanı)
python scripts/curate_dermabench.py build \
    --sources data/dermabench/scin.jsonl data/dermabench/ddi.jsonl \
              data/dermabench/derm1m.jsonl \
    --target-n 1000 --min-malignant 150 \
    --out data/dermabench/dermabench_v1.jsonl

# 3) v1 SCIN görüntülerini indir (DDI zaten lokal)
#    (v1-lite'taki indirme paterni; sadece SCIN yolları)

# 4) Agentic koşum (Colab A100, batch + resume)
python scripts/run_dermagent_subset.py \
    --gold data/dermabench/dermabench_v1.jsonl \
    --dermabench-out data/dermabench/predictions_v1.jsonl \
    --stratified --seed 42 --resume

# 5) Skorla + rapor
python -c "from dermarbiter.evaluation.dermabench import DermAbenchScorer; \
    DermAbenchScorer.from_jsonl('data/dermabench/dermabench_v1.jsonl', \
    'data/dermabench/predictions_v1.jsonl').print_report()"
```

---

## 8. Başarı kriterleri (v1 tamam sayılır)

- [ ] ~1.000 vaka, Fitzpatrick dengeli (VI hariç), ≥150 malign
- [ ] Tüm 8 boyut skorlanıyor, composite + per-dimension rapor
- [ ] Light-vs-dark fairness açığı CI ile raporlanabiliyor
- [ ] Triyaj duyarlılığı ≥150 malign üzerinde stabil
- [ ] Öncelikli ~300-400 vaka clinician-frozen
- [ ] Pre-registration: set eval öncesi donduruldu (seed + hash)

---

## 9. Riskler

| Risk | Etki | Azaltma |
|---|---|---|
| VI=27 yetersiz | Fairness VI'da güçsüz | V+VI birleşik "dark"; v2'de yeni kaynak |
| Compute aşımı | Koşum yarım kalır | Batch + `--resume` + checkpoint |
| API bütçe | Baseline koşulamaz | Abdurrahim key; kademeli koşum |
| Silver≠klinik gerçek | Etiket gürültüsü | Öncelikli alt-küme clinician-frozen |
| Kaynak dağılım kayması | Domain bias | Çok-kaynak + kaynak-kırılımlı rapor |
