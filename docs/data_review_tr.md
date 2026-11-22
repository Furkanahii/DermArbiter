# Veri gözden geçirme ve dondurma

Test bölmesinde koşu yapılabilmesi için verinin bir insan tarafından gözden geçirilip
dondurulması gerekiyor. Kod bunu zorluyor: `data_status=frozen` olmayan veriyle test
koşusu başlamıyor. Bu belge gözden geçirmenin nasıl yapılacağını anlatıyor.

## 1. Örneklem tablosu

```bash
python scripts/make_review_sheet.py data/text_benchmark/scin_text_k30.jsonl --n 50 --out review/scin_text_k30_review.csv
python scripts/make_review_sheet.py data/text_benchmark/medmcqa_skin.jsonl  --n 50 --out review/medmcqa_skin_review.csv
python scripts/make_review_sheet.py data/text_benchmark/medxpertqa_derm.jsonl --n 48 --out review/medxpertqa_derm_review.csv
```

* **Örneklem:** her track'ten sabit bir örneklem alınıyor (case id hash sırası;
  aynı dosya ve aynı N her zaman aynı satırları verir). Satırlar dev ve test
  bölmelerinin ikisinden de geliyor.
* **`shown_to_model`:** modelin gördüğü metnin tam kendisi.
* **`reference`:** MCQ'da doğru şık. SCIN'de dermatolog ağırlıkları; `*` doğru
  sayılan ilk tanıları gösterir, beraberlikte birden fazla olabilir.
* **Gizlilik:** tablolar veri içeriği taşıdığı için git'e eklenmiyor; yerelde
  üretiliyor.

## 2. Her satırda bakılacaklar

| Sütun | Ne zaman `0` yazılır |
|---|---|
| `text_ok` | Metin anlamsız ya da çelişkili, cevabı veya etiketi doğrudan veriyor, ya da görüntü olmadan cevaplanamayacak bir şeye ("resimdeki lezyon...") dayanıyor |
| `reference_ok` | Referans açıkça yanlış; MCQ'da birden fazla şık savunulabilir; SCIN'de referans metinle bariz biçimde uyuşmuyor |
| `issue` | Kısa kod: `leak`, `image`, `ambiguous`, `wrong_key`, `garbled`, `other` |
| `comment` | Serbest not |

Sorun yoksa `1` yazılır. Emin olunamayan satır `1` sayılır ama yorum yazılır.
SCIN'de hasta beyanından tanı koymak zaten zor; "bu metinle ben de tanı koyamazdım"
tek başına `0` gerekçesi değil.

## 3. Kabul ölçütü (veriye bakmadan önce sabitlendi)

* **Kabul:** bir track, örneklemde `text_ok=0` veya `reference_ok=0` olan satır
  oranı %10'u geçmezse kabul edilir.
* **Sistematik sorun:** oran %10'u geçerse ya da `leak` bir kez bile görülürse
  builder düzeltilir, veri yeniden üretilir ve yeni bir örneklemle gözden geçirme
  tekrarlanır.
* **Raporlama:** tek tek sorunlu vakalar veriden elle çıkarılmaz. Sorun oranı
  makalede raporlanır.

## 4. Bilinen özellikler (sorun değil, makalede yazılacak)

* **SCIN etiket beraberlikleri:** 2.154 vakanın 274'ünde (%12,7) beraberlikteki
  ilk etiketlerden biri 30 aday tanının dışında, örneğin "Purpura". Top-1 için
  listedeki eşit ağırlıklı etiket doğru sayıldığından bu vakalar çözülebilir.
  Yumuşak skorun (soft) tavanı ise bu vakalarda 1'in altında kalıyor.
* **SCIN metin kalitesi:** metin hasta beyanı. Bazen çelişkili oluyor; örneğin
  "sağlıklı görünen cilt" deyip "büyüme" bildirilebiliyor.
* **MedMCQA şıkları:** şıklar yeniden karıştırıldı. Orijinal yazım hataları
  (örneğin "Schouleiln") düzeltilmedi.

## 5. Dondurma

Kabulden sonra her track `--data-status frozen --accepted-by "<ad soyad>"` ile yeniden
üretilir (komutlar: [text_benchmark_runbook.md](text_benchmark_runbook.md), Build).
Doldurulmuş tablolar ve sorun oranları gözden geçirmeyi yapanın adı ve tarihiyle
saklanır. Dondurmadan sonra prompt, şema ve ayarlar değişmez; değişirse dev pilotu
tekrarlanır.
