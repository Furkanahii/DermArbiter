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
   7.4'teki kontrol 7B'nin de yetmediğini gösterdi.
2. **Şemsiye etiketlere yığılma.** 200 nihai SCIN cevabının 75'i "Acute and chronic
   dermatitis", 58'i "Acute dermatitis, NOS". Bu iki etiket pilot vakalarında yalnız
   bir kez doğru. Makalede en sık tahminin payı raporlanacak. Aday listeden bu
   etiketlerin çıkarılması bir veri kararı; hocayla konuşulmalı, veriye bakılarak
   tek başına değiştirilmiyor.
3. **Kesinleştirici pilot.** Backbone değişeceği ve 4 ile 5 numaralı düzeltmeler kodu
   değiştirdiği için Kaggle'da son backbone'larla aynı 10 vakalık dev pilot tekrar
   koşulacak; test koşusu ancak ondan sonra.

### 7.4 7B backbone kontrolü (23 Kasım)

**Soru:** 7.3'teki ilk çıkarım "en az 7B" diyordu. 7B, SCIN'de sabit cevap tabanının
üstüne çıkıyor mu?

**Koşu:** Qwen 2.5 7B (Ollama, yerel), `direct` ve `structured`, her track'ten 40 dev
vakası. İlk 10 vaka 3–4B pilotuyla aynı. Ayarlar pilotla aynı: sıcaklık 0,7, tek
tekrar. 160 çağrı, hata yok, maliyet sıfır.

| Track | Kol | top-1 | Sabit taban | 7B − taban | McNemar p | En sık cevap payı |
|---|---|---|---|---|---|---|
| SCIN | `direct` | 2/40 (%5,0) | 10/40 (%25, "Eczema") | −20 [−35, −5] | 0,021 | %50 ("Acute dermatitis, NOS") |
| SCIN | `structured` | 3/40 (%7,5) | 10/40 | −17,5 [−32,5, −2,5] | 0,065 | %49 |
| MedMCQA | `direct` | 28/40 (%70,0) | 12/40 (%30, hep "A") | +40 [+20, +60] | 0,002 | %31 |
| MedMCQA | `structured` | 27/40 (%67,5) | 12/40 | +37,5 [+17,5, +57,5] | 0,003 | %34 |

Köşeli parantezde 5.000 tekrarlı eşleştirilmiş bootstrap ile %95 aralık.

**Bulgular:**

1. **7B, SCIN'de tabanın altında.** Sabit "Eczema" cevabından anlamlı olarak kötü
   (`direct`, p = 0,021). Aynı 10 vakada 3–4B modellerden de iyi değil (7B 0/10;
   Qwen 3B 1/10, Gemma 4B 2/10). Büyüklük tek başına sorunu çözmüyor.
2. **Şemsiye yığılması 7B'de daha güçlü.** 38 nihai cevabın 28'i (%74) iki şemsiye
   etiketten biri. Bu 40 vakada şemsiye etiket yalnız 2 vakada referans.
3. **Şemsiyeleri atlamak da yetmiyor.** Bu bir post hoc, betimleyici sayım: her
   cevabın listesindeki ilk şemsiye olmayan tanı alınırsa top-1 9/40 (%22,5) olur.
   Bu da tabanın (%25) altında. Yani 4b'deki B seçeneği yığılmayı giderir ama
   SCIN'i tek başına tabanın üstüne taşımaz.
4. **MedMCQA sağlıklı.** %70 doğruluk, şık tabanının (%30) çok üstünde, yığılma yok
   (en sık şık %31). 3–4B'ye göre ilk 10 vakada fark yok (7B 6/10; 3B 7/10, 4B 6/10).
5. **Hız:** 7B yerelde vaka başına SCIN ~25 sn, MedMCQA ~15 sn.

**Sınırlılıklar:** n = 40, tek tekrar, tek model ailesi, gözden geçirilmemiş dev
verisi. Yine de SCIN farkı yönü ve büyüklüğüyle pilottaki tabloyu doğruluyor.

**Çıkarımlar** (karar hocanın; test bölmesine bakılmadan verilmeli):

* **Track önceliği.** MedMCQA ve MedXpertQA birincil track olmaya aday. SCIN ancak
  bir modelin dev bölmesinde tabanı açıkça geçtiği gösterilirse birincil kalmalı.
  Aksi halde ikincil analiz ya da "görüntüsüz hasta beyanının sınırı" bulgusu olarak
  raporlanır. Bu kendi başına yayımlanabilir bir sonuç: kollar arası fark aranan
  track'te önce modelin işe yarar bir sinyal alması gerekiyor.
* **Daha güçlü ücretsiz backbone denemesi.** Kaggle'daki kesinleştirici dev pilotunda
  SCIN için bir 12–14B model (örneğin Gemma 3 12B veya Qwen 2.5 14B) aynı 40 vakada
  `direct` ile denenebilir. Tabanı geçmezse karar netleşir.
* **Prompt (4b, C seçeneği).** "En özgül tanıyı seç" eklemesi yığılmayı azaltabilir,
  ama 3. bulguya göre doğruluğu tabanın üstüne taşıması beklenmiyor.

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

## 9b. Proje değerlendirmesi ve riskler (23 Kasım 2026)

**Sağlam olanlar.** Tasarım hakem itirazlarının çoğunu önceden karşılıyor:

* **Adil karşılaştırma:** eşit ve sayılmış çağrı bütçeleri, ortak tohumlar, bütün
  kollara aynı şema.
* **Önceden sabitlenmiş analiz:** beş karşılaştırma ve Holm düzeltmesi.
* **Payda:** planlanan tüm vakalar paydada.
* **Kodla zorlanan kurallar:** test kapısı ve manifest hash'leri.
* **Tekrarlanabilir veri:** hash'li veri üretimi.
* **Erken yakalanan hatalar:** pilot test bölmesine dokunmadan beş gerçek hata
  buldu, hepsi düzeltildi.

Harcama şimdiye kadar sıfır.

**Riskler, önem sırasıyla:**

1. **SCIN'de taban sorunu (gerçekleşti).** Görüntüsüz hasta beyanı zor bir görev ve
   sabit cevap tabanı %25. 3–4B modeller tabanda kaldı; 7B kontrolünde Qwen 2.5 7B
   tabanın anlamlı olarak altında (%5, 7.4). Önerilen önlem: MedMCQA/MedXpertQA
   birincil, SCIN ikincil analiz. Kaggle'da 12–14B bir model tabanı geçerse SCIN
   yeniden birincil olabilir. Karar test bölmesine bakılmadan verilir.
2. **Güç.** 600 SCIN vakasıyla saptanabilir fark 4–6 puan. Literatürde harness
   etkileri çoğu zaman 1–5 puan. Kaggle'da ilk iş verim ölçülmeli. Verim yeterliyse
   SCIN testinin tamamı (1.737) koşulur ve saptanabilir fark 2–4 puana iner.
3. **İki küçük açık model ailesi.** Bu durum S4'ün (sıralama backbone'a göre korunuyor
   mu?) genellenebilirliğini sınırlıyor. Kapalı, güçlü bir model ancak ayrı bir bütçe
   kararıyla eklenebilir.
4. **Tek tekrar.** Koşudan koşuya kararlılık henüz ölçülmedi. Analiz aracı hazır
   (`repeat_agreement`); dev pilotu Kaggle'da 2 tekrarla koşulmalı.
5. **Açık veri kararları.** Şemsiye etiketler ([data_review_tr.md](data_review_tr.md)
   4b) ve insan gözden geçirmesi tamamlanmadan test bölmesi açılamaz.
6. **Adlandırma.** Makaledeki `dermarbiter` kolu, DermArbiter'in persona ve moderatör
   tasarımının araçsız bir uyarlaması; depodaki LangGraph/araç sisteminin kendisi
   değil. Makale bunu açıkça söylüyor; hakem sorusuna hazır olmak gerekiyor.
7. **Kirlenme.** MedMCQA'nın açık train bölmesi ön eğitim verisinde olabilir. Bu
   sınırlılık olarak yazılıyor; mutlak doğruluk değil, kollar arası farklar
   yorumlanıyor.

## 10. Takvim ve durum

Takvim [research_decisions_v2_tr.md](research_decisions_v2_tr.md) 7. bölümde.

| Durum | İş |
|---|---|
| Tamam | Pipeline düzeltmeleri, veri builder'ları, 10 kol, istatistik, literatür raporu |
| Tamam | Yerel pilot (planlanan 1–6 Aralık yerine 22–23 Kasım); paralel runner; Kaggle notebook'u |
| Tamam | 7B backbone kontrolü (7.4): SCIN tabanın altında, MedMCQA sağlıklı |
| Sırada | Track önceliği kararı (7.4, hocayla); veri gözden geçirme ve dondurma (insan onayı); Kaggle'da son backbone'larla dev pilot; ana koşular (7–13 Aralık) |
| Sonra | Görüntü aşaması (2. aşama), guideline-RAG ve heterojen panel kolları (ek analiz) |
