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
  zorunlu sayı yapıyor, gerekçeyi 1000 karakterle sınırlıyor. Bu kısıtların üçü de
  pilotta görülen gerçek hatalardan sonra eklendi (bkz. 7. bölüm).
* **Payda:** planlanan tüm vakalar paydada. Eksik, hatalı ve çekimser cevaplar doğru
  sayılmıyor. Oylama beraberliği çekimser kalıyor.
* **Kayıt:** her koşu için manifest tutuluyor: kod ve şema hash'i, istenen ve dönen
  model sürümü, zaman aşımı, `publication_eligible` bayrağı.

## 6. Ölçümler ve istatistik

* **Ölçümler:** top-1 doğruluk (ana ölçüt), top-3 ve dermatolog ağırlıklı yumuşak
  skor (SCIN), cevap kapsamı, vaka başına çağrı, token ve süre, faydalı/zararlı
  revizyon sayıları.
* **Fark testleri:** vaka düzeyinde eşleştirilmiş bootstrap %95 güven aralığı ve exact
  McNemar testi; birincil aileye Holm düzeltmesi.
* **Varyans ayrıştırması:** vaka × model × harness kareler toplamı (S4).

**Örneklem büyüklüğü.** Kaba hesap: eşleşmeyen çift oranı ~%20 ve Holm ile α=0,01
alınırsa, %80 güçle saptanabilecek en küçük fark yaklaşık 1.737 SCIN test vakasında
3,7 puan, 600 vakada 6,2 puan, 300 vakada 8,8 puan.

## 7. Pilot (yerel, ücretsiz)

* **Kurulum:** Ollama, Apple M2 / 16 GB. Backbone'lar `gemma3:4b` ve `qwen2.5:3b`.
  Her track'ten 10 dev vaka, 10 kolun hepsi; model başına track başına 420 çağrı.
* **Amaç:** sonuç değil sağlamlık. Format hataları, çekimserlik, token ve süre
  ölçülüyor.

Pilotun şimdiden yakaladığı ve düzelttiği üç hata:

1. **Boş JSON:** düz JSON modunda Gemma `{}` döndürüyordu (`direct`'te 10/10
   geçersiz). Çözüm: vaka bazlı JSON şeması; sonrasında 20 vakada 0 hata.
2. **Gerekçe döngüsü:** bir SCIN vakasında Gemma gerekçe metninde döngüye girdi,
   1024 token sınırına kadar yazdı ve 120 sn zaman aşımına düştü. Her çok çağrılı
   kolda aynı vaka düştüğü için kollar arası farkı sistematik biçimde bozuyordu.
   Çözüm: gerekçeye 1000 karakter sınırı ve kayıtlı `--timeout` seçeneği.
3. **Eksik güven:** Qwen tanı verirken güveni `null` bırakıyordu ve cevap geçersiz
   sayılıyordu. Çözüm: güven şemada zorunlu sayı; çekimser cevaplarda yok sayılıyor.

Şema değiştiği için ilk Gemma koşusu (`local_pilot_v0_nocap`) arşivlendi ve pilot iki
model için baştan başlatıldı. Sonuç tablosu bu bölüme eklenecek.

> **Pilot sonuçları:** koşu sürüyor.

## 8. Ana deneyler için hesaplama planı (maliyetsiz)

**Yerel makine yetmiyor.** Bu makinede 4B bir model çağrı başına ~20 sn alıyor.
SCIN testinin tamamı model başına 1.737 × 42 ≈ 73 bin çağrı ediyor; yerelde yüzlerce
saat sürer.

**Önerilen yol:**

* **Donanım:** Kaggle (haftada ~30 saat ücretsiz GPU) veya Colab T4 üzerinde vLLM
  ile açık ağırlıklı iki backbone, örneğin Qwen 2.5 7B ve Gemma 3 12B'nin
  kuantize sürümleri. vLLM OpenAI uyumlu olduğu için mevcut
  `openai_compatible` sağlayıcı değişmeden kullanılıyor.
* **Paralellik:** runner'a vakaları paralel gönderen bir seçenek eklenecek. vLLM
  istekleri toplu işlediği için verimi kat kat artırır.
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
| Sürüyor | Yerel pilot (planlanan 1–6 Aralık yerine şimdiden) |
| Sırada | Paralel runner; Kaggle/Colab vLLM kurulumu; veri dondurma ve onay; ana koşular (7–13 Aralık) |
| Sonra | Görüntü aşaması (2. aşama), guideline-RAG ve heterojen panel kolları (ek analiz) |
