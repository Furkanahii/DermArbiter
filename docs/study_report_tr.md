# Çalışma raporu: metin tabanlı dermatoloji vakalarında ajan harness karşılaştırması

Taslak, 22 Kasım 2026. Hocanın 2 numaralı teslimi. Literatür raporu ayrı dosyada:
[literature_report_2026_tr.md](literature_report_2026_tr.md). Komutlar ve teknik
ayrıntı: [text_benchmark_runbook.md](text_benchmark_runbook.md).

Önerilen başlık: *An improved agentic harness for medical cases: a dermatology use case
with call-matched comparisons on text-only patient reports*.

## 1. Çerçeve

Makale "geliştirdiğimiz model hepsinden iyi" demiyor. Aynı model, aynı vakalar ve
açıkça sayılmış çağrı bütçeleri altında bir **yaklaşımlar kümesini** (harness kolları)
birbiriyle karşılaştırıyor. Bu, hocanın örnek verdiği iki çalışmanın (arXiv 2009.06366,
2110.12270) "yöntem ızgarası + tek ana tablo" kurgusunun ajan sistemlerine uyarlaması.
DermArbiter'in rol tabanlı tartışma ve moderatör tasarımı bu kümenin bir üyesi; diğer
kollar onun her parçasını ayrı ayrı test eden kontroller.

Bu aşama yalnız metin. Görüntü 2. aşamada aynı SCIN vakalarına eklenecek; böylece
"görüntü eklemek harness sıralamasını değiştiriyor mu?" sorusu aynı vakalarda sorulabilir.

## 2. Araştırma soruları

* **S1.** Aynı backbone'da harness seçimi doğruluğu ne kadar değiştiriyor? Eşit çağrı
  bütçesinde fark kalıyor mu?
* **S2.** Kazanç ek hesaplamadan mı (oylama), ajanlar arası iletişimden mi (tartışma)
  geliyor? Kontroller: `vote6` ve `isolated_refine` ile `debate`, hepsi 6 çağrı.
* **S3.** DermArbiter'in rol personaları ve moderatörü homojen tartışmanın üstüne bir
  şey katıyor mu? (`role_vote`, `role_debate`, `dermarbiter`)
* **S4.** Sonuçtaki varyansın ne kadarı vakadan, modelden, harness'ten ve model ×
  harness etkileşiminden geliyor? Harness sıralaması backbone değişince korunuyor mu?
  (MedicalHarness, arXiv 2610.05778 ile doğrudan kıyaslanabilir analiz.)
* **S5.** Bulgular görev türüne göre değişiyor mu: bilgi sorusu (MCQ) ile hasta beyanından
  ayırıcı tanı?

## 3. Veri

| Track | Görev | Kaynak ve lisans | Vaka (dev / test) | Sabit cevap baseline (test) | Rastgele (test) |
|---|---|---|---|---|---|
| `scin_text_k30` | 30 aday tanı arasından ≤3 tanı sırala | SCIN, hasta beyanı metadata, görüntüsüz; dermatolog panel etiketi | 417 / 1.737 | 25,5% ("Eczema") | 4,4% |
| `medmcqa_skin` | Tek doğru şıklı MCQ | MedMCQA Skin alt kümesi, Apache-2.0 | 257 / 1.012 | 27,1% | 25,0% |
| `medxpertqa_derm` | 10 şıklı zor MCQ | MedXpertQA dermatoloji, MIT | 0 / 48 | 20,8% | 10,0% |

Veri kararları:

* **SCIN:** yalnız dermatolog etiketli vakalar alındı. En sık 30 tanı aday listesi
  oldu. Referansı listenin dışında kalan 509 vaka ve ikiden az alan bildirilmiş
  398 vaka çıkarıldı. Dermatolog ağırlıkları toplamı 1 olacak şekilde normalize
  edildi. Eşit ağırlıklı ilk tanıların hepsi doğru sayılıyor (697 vaka).
* **MedMCQA:** çok cevaplı (491), boş veya tekrar eden şıklı (7) ve görüntüye
  dayalı (4) sorular çıkarıldı. Şıklar soru başına deterministik olarak
  karıştırıldı, çünkü orijinal anahtar %35–40 oranında "A"ya kayıktı.
* **Bölme:** dev/test bölmesi, görünen metne göre gruplanmış deterministik hash ile
  yapıldı; aynı metin iki bölmeye düşemiyor. Kaynak revizyonu ve dosya hash'leri
  `data/text_benchmark/*.report.json` içinde.
* **Durum:** tüm veri `pilot_candidate`. Test bölmesi, veri dondurma kaydı
  (`--data-status frozen --accepted-by ...`) olmadan çalıştırılamıyor; bunu kod
  zorluyor.

## 4. Harness kolları

| Kol | Çağrı | Neyi izole ediyor |
|---|---:|---|
| `direct` | 1 | ham backbone |
| `structured` | 1 | yapılandırılmış prompt |
| `self_refine` | 3 | aynı ajana ek hesaplama |
| `vote3` / `vote6` | 3 / 6 | yalnız örnekleme (çoğunluk / Borda) |
| `isolated_refine` | 6 | tartışma topolojisi, akran bilgisi yok |
| `debate` | 6 | homojen akran iletişimi |
| `role_vote` | 3 | DermArbiter personaları (uzman / genel / şüpheci), iletişim yok |
| `role_debate` | 6 | personalar + isimli akranları görerek bir eşzamanlı revizyon turu |
| `dermarbiter` | 7 | `role_debate` + skorlanan kararı moderatör veriyor |

Önceden belirlenmiş birincil karşılaştırmalar (model başına Holm düzeltmesi):
`debate − isolated_refine`, `debate − vote6`, `role_vote − vote3`,
`role_debate − debate`, `dermarbiter − role_debate`. Her kolun `direct`'e göre farkı
ikincil analiz.

## 5. Adillik kontrolleri

* **Aynı girdi ve tohumlar:** her kol aynı vakaları, aynı görünen metni ve aynı
  çağrı tohumlarını kullanıyor.
* **Kapalı ek kaynaklar:** araç, retrieval, görüntü ve gizli yeniden deneme yok.
  Altın etiket benzeri alanlar prompttan filtreleniyor.
* **Aynı çıktı şeması:** her kola aynı vaka bazlı JSON şeması veriliyor (constrained
  decoding). Şema cevabı geçerli şıklarla veya aday tanılarla sınırlıyor, güveni
  zorunlu sayı yapıyor, gerekçeyi 1000 karakterle sınırlıyor. Bu kısıtlar pilotta
  görülen gerçek hatalardan sonra eklendi (bkz. 7. bölüm).
* **Payda:** planlanan tüm vakalar paydada. Eksik, hatalı ve çekimser cevaplar doğru
  sayılmıyor.
* **Toplama:** MCQ'da çoğunluk oyu; beraberlik çekimser. SCIN'de Borda (3/2/1 puan);
  ilk sıradaki puan beraberliği birinci sıra oy sayısıyla kırılıyor, o da eşitse
  çekimser. Ajanların aynı tanıyı listede tekrar etmesi hata sayılmıyor; tekrar
  atılıyor, sıra korunuyor (top-1 değişmez).
* **Kayıt:** her koşu için manifest tutuluyor: kod ve şema hash'i, istenen ve dönen
  model sürümü, zaman aşımı, `publication_eligible` bayrağı.

## 6. Ölçümler ve istatistik

* **Ölçümler:** top-1 doğruluk (ana ölçüt), top-3 ve dermatolog ağırlıklı yumuşak
  skor (SCIN), cevap kapsamı, vaka başına çağrı, token ve süre, faydalı/zararlı
  revizyon sayıları.
* **Fark testleri:** vaka düzeyinde eşleştirilmiş bootstrap %95 güven aralığı ve exact
  McNemar testi; birincil aileye Holm düzeltmesi.
* **Varyans ayrıştırması:** vaka × model × harness kareler toplamı (S4).

**Örneklem büyüklüğü.** Eşleşmeyen çift oranı p ve Holm ile α=0,01 alınırsa, %80
güçle saptanabilecek en küçük fark ≈ 3,42·√(p/n). Pilotta beş birincil karşılaştırmada
gözlenen oran %7 (200 çiftte 14; SCIN %3, MedMCQA %11). Küçük modeller SCIN'de tabana
yakın olduğu için bu oran büyük modellerde artacaktır; planlamada temkinli değer
%20 kullanılıyor.

| n | p = %7 | p = %20 |
|---|---|---|
| 1.737 (SCIN testin tamamı) | 2,2 puan | 3,7 puan |
| 600 | 3,7 puan | 6,2 puan |
| 400 | 4,5 puan | 7,6 puan |

## 7. Pilot (yerel, ücretsiz)

* **Kurulum:** Ollama, Apple M2 / 16 GB. Backbone'lar `gemma3:4b` ve `qwen2.5:3b`.
  Her track'ten 10 dev vaka, 10 kolun hepsi; model başına track başına 420 çağrı.
* **Amaç:** sonuç değil sağlamlık. Format hataları, çekimserlik, token ve süre
  ölçülüyor.

Pilot 22 Kasım 16:31'de başladı, 23 Kasım 01:15'te bitti: 40 koşu (2 model × 2 track ×
10 kol), 400 vaka-koşusu, 1.673 çağrı, çağrı başına ~19 sn. Bütün koşular aynı kod ve
şema hash'iyle tamamlandı.

### 7.1 Pilotun yakaladığı ve düzeltilen hatalar

İlk üçü pilotun ilk denemesinde görüldü; şema değiştiği için o koşu
(`local_pilot_v0_nocap`) arşivlendi ve pilot baştan başlatıldı. Son ikisi asıl pilotta
görüldü.

1. **Boş JSON:** düz JSON modunda Gemma `{}` döndürüyordu (`direct`'te 10/10
   geçersiz). Çözüm: vaka bazlı JSON şeması.
2. **Gerekçe döngüsü:** Gemma bir SCIN vakasında gerekçede döngüye girip zaman
   aşımına düştü; aynı vaka her çok çağrılı kolda düştüğü için kollar arası farkı
   sistematik bozuyordu. Çözüm: gerekçeye 1000 karakter sınırı, kayıtlı `--timeout`.
3. **Eksik güven:** Qwen tanı verirken güveni `null` bırakıyordu. Çözüm: güven
   şemada zorunlu sayı; çekimser cevaplarda yok sayılıyor.
4. **Tekrarlanan tanı:** Qwen 3 çağrıda listeye aynı tanıyı iki kez yazdı
   (`["Acne", "Acne"]`). Ollama şemadaki tekrarsızlık kuralını uygulamıyor ve parser
   bunu geçersiz sayıyordu: 3 vaka düştü. Çözüm: tekrar atılıyor, sıra korunuyor,
   sayısı kaydediliyor (`duplicates_dropped`). Bu vakalar yeni model çağrısı olmadan
   yeniden puanlanamaz; tabloda hata olarak duruyor.
5. **Borda beraberliği:** iki ajanın ilk sıraya koyduğu tanı, üç ajanın da ikinci
   sıraya koyduğu tanıyla puanda eşitleniyordu (3+3 = 2+2+2) ve panel çekimser
   kalıyordu. Toplam 26 çekimser sonucun 15'i buydu ve en çok rol kollarını
   vuruyordu, çünkü personalar daha farklı cevap veriyor. Kişi bazında
   çekimserlik düşük: uzman %4,0, genel %2,5, şüpheci %4,5. Çözüm: puan
   beraberliği birinci sıra oy sayısıyla kırılıyor. Toplanan cevap hiçbir prompta
   girmediği için pilot model çağırmadan, birebir yeniden puanlandı
   (`scripts/reaggregate_runs.py`): 13 nihai cevap çekimserden tanıya döndü; hiçbiri doğru
   değildi, top-1 değişmedi, kapsam arttı.

Kalan tek açıklanamayan hata: Gemma bir SCIN çağrısında gerekçenin içine kıvrık
tırnak yazdı, ardından anlamsız metin üretti ve sunucu yanıtı bitiş nedeni
vermeden kesti (`incomplete_generation`). Aynı çağrı `role_debate` ile `dermarbiter`'da
ortak olduğu için iki kolda birlikte görülüyor; bu ikisinin karşılaştırmasını bozmuyor.

### 7.2 Sonuçlar (10 dev vaka; sağlamlık testi, sonuç değil)

Top-1 doğruluk / kapsam, %. SCIN güncel toplama kuralıyla yeniden puanlandı.

| Kol | SCIN Gemma 3 4B | SCIN Qwen 2.5 3B | MedMCQA Gemma | MedMCQA Qwen |
|---|---|---|---|---|
| `direct` | 20 / 100 | 10 / 100 | 60 / 100 | 70 / 90 |
| `structured` | 20 / 100 | 10 / 100 | 60 / 100 | 60 / 90 |
| `self_refine` | 40 / 100 | 0 / 80 | 60 / 100 | 60 / 90 |
| `vote3` | 20 / 100 | 10 / 100 | 50 / 100 | 60 / 90 |
| `vote6` | 20 / 100 | 0 / 90 | 50 / 90 | 60 / 90 |
| `isolated_refine` | 20 / 100 | 10 / 100 | 50 / 100 | 60 / 80 |
| `debate` | 20 / 100 | 10 / 100 | 60 / 100 | 70 / 90 |
| `role_vote` | 10 / 100 | 10 / 90 | 60 / 90 | 60 / 90 |
| `role_debate` | 10 / 90 | 10 / 100 | 50 / 100 | 70 / 100 |
| `dermarbiter` | 10 / 90 | 10 / 100 | 70 / 100 | 60 / 100 |

* **Hatalar:** 400 vaka-koşusunun 5'i (%1,25): yukarıdaki 3 tekrar ve 1 kesik çağrı
  (2 kolda).
* **Fark testleri:** 20 önceden belirlenmiş karşılaştırmanın hiçbiri anlamlı değil
  (Holm sonrası hepsi p = 1). En büyük fark MedMCQA Gemma'da
  `dermarbiter − role_debate` +20 puan (2/0 uyuşmayan vaka). n = 10 ile beklenen.
* **Maliyet:** `dermarbiter` vaka başına `direct`'in 7 katı çağrı, 13–15 katı token
  (MedMCQA ~4.400, SCIN ~7.000) ve 7–12 katı süre (123–221 sn) harcıyor.
* **Varyans ayrıştırması** (10 vaka × 2 model × 10 kol): SCIN'de vaka %64,7, harness'e
  atfedilebilen toplam %14,5 (harness ana etkisi %0,9, model × harness %2,8).
  MedMCQA'da vaka %45,0, vaka × model %34,2, harness toplamı %14,0. n = 10 ile
  yorumlanmıyor; yalnız analiz zincirinin uçtan uca çalıştığını gösteriyor.
* **Uyuşmazlık oranı:** %7 (bkz. 6. bölüm, örneklem büyüklüğü).

### 7.3 Ana koşular için çıkarımlar

1. **3–4B modeller SCIN'de tabanda.** Bu 10 vakada sabit "Eczema" cevabı %20 alıyor;
   Gemma `direct` de %20, Qwen %10. Harness farkını görmek için tabanın üstünde bir
   backbone gerekiyor. Ana koşularda en az 7B (örneğin Qwen 2.5 7B, Gemma 3 12B).
2. **Şemsiye etiketlere yığılma.** 200 nihai SCIN cevabının 75'i "Acute and chronic
   dermatitis", 58'i "Acute dermatitis, NOS". Bu iki etiket pilot vakalarında yalnız
   bir kez doğru. Makalede en sık tahminin payı raporlanacak. Aday listeden bu
   etiketlerin çıkarılması bir veri kararı; hocayla konuşulmalı, veriye bakılarak
   tek başına değiştirilmiyor.
3. **Kesinleştirici pilot.** Backbone değişeceği ve 4 ile 5 numaralı düzeltmeler kodu
   değiştirdiği için Kaggle'da son backbone'larla aynı 10 vakalık dev pilot tekrar
   koşulacak; test koşusu ancak ondan sonra.

## 8. Ana deneyler için hesaplama planı (maliyetsiz)

**Yerel makine yetmiyor.** Bu makinede 4B bir model çağrı başına ~20 sn alıyor.
SCIN testinin tamamı model başına 1.737 × 42 ≈ 73 bin çağrı ediyor; yerelde yüzlerce
saat sürer.

**Önerilen yol:**

* **Donanım:** Kaggle (haftada ~30 saat ücretsiz GPU) veya Colab T4 üzerinde vLLM
  ile açık ağırlıklı iki backbone, örneğin Qwen 2.5 7B ve Gemma 3 12B'nin
  kuantize sürümleri. vLLM OpenAI uyumlu olduğu için mevcut
  `openai_compatible` sağlayıcı değişmeden kullanılıyor.
* **Paralellik:** runner `--workers` ile vakaları paralel gönderiyor; istemler,
  tohumlar ve kayıt sırası değişmiyor (testle doğrulandı). Kaggle notebook'u
  (`notebooks/08_text_benchmark_kaggle.ipynb`) bunu kullanıyor.
* **Örneklem:** test bölmesinden sabit hash sıralı alt küme, örneğin SCIN 600 +
  MedMCQA 400 + MedXpertQA 48, yani model başına ~44 bin çağrı. Bu ölçekte
  saptanabilir fark 6 puan civarı; makalede açıkça yazılacak.
* **Ücretli API:** yalnız ayrı bir bütçe kararı verilirse güçlü bir kapalı model eklenir.

## 9. Sınırlılıklar

* **Hasta beyanı metni:** SCIN metni hasta beyanı olduğu için tanı görüntüsüz
  zor ve gürültülü. Sabit cevap baseline'ı %25 düzeyinde ve sonuçlar buna göre
  okunmalı.
* **Çağrı ve token eşitliği:** eşit çağrı sayısı eşit token anlamına gelmiyor;
  token tüketimi ayrıca raporlanıyor.
* **Küçük backbone'lar:** küçük modellerde harness etkisi büyük modellerdekinden
  farklı olabilir. S4 bu yüzden en az iki backbone gerektiriyor.
* **Etiket belirsizliği:** MedMCQA etiket gürültüsü içeriyor. SCIN etiketleri panel
  ağırlıklı ve beraberlikli.

## 10. Takvim ve durum

Takvim [research_decisions_v2_tr.md](research_decisions_v2_tr.md) 7. bölümde.

| Durum | İş |
|---|---|
| Tamam | Pipeline düzeltmeleri, veri builder'ları, 10 kol, istatistik, literatür raporu |
| Tamam | Yerel pilot (planlanan 1–6 Aralık yerine 22–23 Kasım); paralel runner; Kaggle notebook'u |
| Sırada | Veri gözden geçirme ve dondurma (insan onayı); Kaggle'da son backbone'larla dev pilot; ana koşular (7–13 Aralık) |
| Sonra | Görüntü aşaması (2. aşama), guideline-RAG ve heterojen panel kolları (ek analiz) |
