# DermArbiter metin tabanlı deney protokolü taslağı

21 Kasım 2026. Durum: önerilen protokol; ön kayıt yapılmış veya gerçek deney tamamlanmış değildir. Bu belge, 15 Kasım danışman geri bildirimi doğrultusunda ilk makalenin kapsamını tanımlar. Depodaki görsel DermArena protokolü ikinci aşama için tarihsel tasarım olarak korunur.

## Amaç ve araştırma soruları

Amaç, dermatoloji olgularında aynı modelin farklı çalışma düzeneklerini karşılaştırmaktır. Burada harness; istem şablonları, bilgi getirme araçları, ajanlar arası iletişim, karar birleştirme, hata yönetimi ve bütçe sınırlarının bütünüdür. Model ağırlıkları eğitilmez. “Training-free” ifadesi development vakalarında istem veya eşik seçilmediği anlamına gelmez; bu seçimler açıkça açıklanır.

1. Yapılandırılmış çıktı ve klinik kontrol listesi, doğrudan yanıta göre ne değiştiriyor?
2. Aynı metin corpus'undan RAG eklemek doğruluk ve maliyeti nasıl etkiliyor?
3. Çok ajanlı tartışma, aynı modelden bağımsız örneklerin oylanmasından daha yararlı mı?
4. Bu etkiler en az iki temel modelde benzer mi?
5. Hangi yöntem daha sık tamamlanıyor, ne zaman yanlışken yüksek güven veriyor, hangi vaka türlerinde başarısız oluyor?

## Görev ve izin verilen bilgi

Birincil görev, tanı öncesinde mevcut metinsel vaka öyküsünden sıralı diferansiyel tanı üretmektir. Metin; belirtiler, süre, dağılım, öykü ve varsa değerlendirme anında bilinen muayene bulgularını içerir. Her koşula aynı metin verilir. Yanıt en fazla üç standartlaştırılabilir tanı, top-1 güven değeri, kısa kanıt gerekçesi ve gerektiğinde abstention içerir.

Görüntü dosyası, URL'si, gömmesi, görsel araç sonucu veya test tanısını açığa çıkaran dosya adı model girdisine dahil edilmez. RAG varsa sorgu yalnız görünür vaka metninden türetilir. Tanı ve kaynak makale başlığı model sorgusuna otomatik eklenmez. Aynı olgunun kaynak makalesinin RAG içinde bulunması engellenir.

Native clinical text ile klinisyen yazımı vignette ana veri adaylarıdır. Görüntüden üretilmiş caption ve etiket temelinde üretilmiş sentetik vignette, ana klinik başarı tablosuna karıştırılmaz. Görseli görmeden cevaplanamayacak soru, birincil text-only kohorta alınmaz. “Görüntü yokken yetersiz bilgiye tepki” ayrı bir değerlendirme olabilir.

Açık uçlu tanı ile çoktan seçmeli tıbbi sorular ayrı görevlerdir; tek accuracy tablosunda birleştirilmez. Birincil kapsam tüm dermatoloji mi, pigmente lezyonlar mı, nadir hastalıklar mı olacak veri bulunabilirliğine göre dondurulmalıdır. Veri yalnız lezyonlardan oluşursa başlık ve sonuçlar bütün dermatolojiye genellenmez.

## Karşılaştırılacak altı koşul

| Kod | Koşul | Çağrı düzeni | Araç |
|---|---|---|---|
| A0 | Doğrudan tek model | Bir çağrı, ortak cevap şeması | Yok |
| A1 | Yapılandırılmış tek model | Bir çağrı, önceden belirlenmiş klinik kontrol listesi | Yok |
| A2 | Tek model ve RAG | Sabit getirme, tek karar çağrısı | Dondurulmuş guideline metni |
| A3 | Bağımsız örnekleme | Aynı modelden k bağımsız yanıt, önceden tanımlı oylama | Yok |
| A4 | Çok ajanlı tartışma | Aynı modelden farklı roller, bir tartışma turu ve yapılandırılmış son karar | Yok |
| A5 | Çok ajanlı tartışma ve RAG | A4 ile aynı akış, A2 ile aynı getirme ayarları | Aynı guideline corpus'u |

A1/A2/A4/A5 yaklaşık 2×2 RAG × tartışma karşılaştırmasına olanak verir; role promptlarının farkı da yöntemin parçasıdır. A3, çok çağrının getirisi ile tartışmanın getirisini ayıran güçlü kontroldür. A0 ve A1 aynı JSON şemasını kullanır; format başarısızlığı yöntem farkı gibi karıştırılmamalı.

İlk ana deneyde bütün roller aynı temel modelden kurulmalıdır. Heterojen Gemini/MedGemma/Qwen paneli model bileşimi ile iletişim etkisini karıştırdığı için ikincil deneydir. En az iki sabit model sürümünde koşullar tekrarlanır. “En güncel/en güçlü” iddiası yerine kesin provider/model ID, tarih ve parametreler kaydedilir. Model seçimi erişim ve pilot maliyetiyle kararlaştırılır.

Tartışmasız rol ensemble, ikinci tartışma turu, skeptic çıkarma ve early-exit yalnız ana koşullar çalıştıktan sonra ablation olur. Vision modelleri, yeni fine-tuning, 11 soru tipi, geniş yönetim/tedavi değerlendirmesi ve yeni çok boyutlu benchmark iddiası ilk makalenin kritik yoluna eklenmez.

## Veri hazırlama ve kör inceleme

1. Kaynak, lisans/yeniden dağıtım şartları, olgu kimliği ve etiketin elde edilme biçimi kaydedilir. Bir makalenin açık erişimli olması, içindeki her veri unsurunun sınırsız yeniden dağıtılabileceğini tek başına göstermez.
2. Tanı öncesi vaka metni ile gold cevap ayrılır. Başlıklar, tanı cümleleri, patoloji sonucu, ayırt edici caption'lar ve tedavi yanıtından tanı çıkarımı uzman tarafından gözden geçirilir. Klinik görevde meşru ipucu ile doğrudan cevap sızıntısı ayırt edilir.
3. Aynı hasta, olgu serisi, makale veya çok benzer vaka tek grup altında tutulur. Aynı gruptan dev/test'e dağıtım yapılmaz. Yakın kopyalar metin benzerliği ve manuel inceleme ile kontrol edilir.
4. En az bir yetkin klinik değerlendirici tanıyı ve metnin tek başına görev için yeterliliğini kontrol eder. Uygun bir altkümede ikinci bağımsız değerlendirme, uyuşmazlık çözümü ve uyum raporu önerilir. Nihai örneklem, gerçek klinik kapasiteye bağlıdır.
5. Model sonuçları görülmeden etiket, kabul edilen eşanlamlılar ve scoring rubric dondurulur. Vaka kaynakları ve RAG corpus'u arasında örtüşme denetlenir.
6. Test seti yalnız son koşullar için kullanılır. Test sonuçlarına bakılarak yöntem seçilirse aynı test üzerinde doğrulayıcı iddia kurulmaz.

20–30 development vakası ve 150–250 test vakası başlangıç planlama aralığıdır; güç analizi sonucu değildir. Örneğin 200 bağımsız vakada p=0.5 için tek oran standart hatası yaklaşık 0.035'tir; normal yaklaşımda %95 aralık yarı genişliği yaklaşık 6.9 yüzde puandır. Yöntem farkının kesinliği eşleştirilmiş uyuşmazlıklara ve kaynak kümelenmesine bağlıdır. Küçük farkları kanıtlama sözü verilmemeli.

### Bir vaka için önerilen veri sözleşmesi

```json
{
  "case_id": "case-0001",
  "source_id": "article-or-patient-group-001",
  "source": "curated_source",
  "source_url": "https://example.org/source",
  "license": "verified terms recorded here",
  "split": "dev",
  "modality": "text",
  "text_origin": "native_clinical_text",
  "clinical_history": "Diagnosis-free presentation text goes here.",
  "query": "Return up to three ranked diagnoses and your top-1 confidence.",
  "patient_context": {"age": 40, "location": "forearm"},
  "ground_truth": {"diagnosis_label": "REFERENCE_LABEL"},
  "annotation_status": "frozen",
  "reviewer_id": "reviewer-pseudonym"
}
```

Bu örnek yalnız şemadır, gerçek vaka değildir. Görünür model girdisi `clinical_history`, `query` ve izinli `patient_context` alanlarıdır. Gold ve source alanları değerlendiricide tutulur. Mevcut eski runner'lar bu sözleşmeyle henüz çalışmaz.

Başlangıç denetimi: `python scripts/validate_text_benchmark.py cases.jsonl --output audit.json`. Script yalnız yapısal denetim yapar; “geçti” sonucu, klinik geçerlilik veya yayın uygunluğu belgesi değildir. Çıktıdaki hash, tam dosya baytlarını tanımlar; insan kontrolünün yerine geçmez.

## Ölçümler

| Ölçüt | Tanım | Kullanım |
|---|---|---|
| Top-1 accuracy | Doğru ilk tanı / dondurulmuş tüm test vakaları | Birincil |
| Top-3 accuracy | Gold tanının ilk üçte bulunması / tüm test vakaları | İkincil |
| Macro-F1 ve balanced accuracy | Önceden sabitlenmiş etiket ontolojisine göre | Sınıfların yeterli desteği varsa |
| Valid completion rate | Geçerli, skorlanabilir yanıt / tüm vakalar | Operasyonel güvenilirlik |
| Hata ve abstention oranları | Ayrı hata kategorileri ve bilinçli kararsızlık | Accuracy'nin açıklaması |
| ECE | Son top-1 kararına bağlı güven ve doğruluk uyumu | Güven alanı doğrulandıktan sonra |
| Binary correctness Brier | (Top-1 güven − doğru/yanlış)^2 ortalaması | Çok sınıflı Brier'dan açıkça ayrılır |
| Süre ve kullanım | Çağrı, retry, giriş/çıkış tokenı, medyan/p95 süre, maliyet | Ana sonuçlarla birlikte |
| Citation desteği | Atıfın ilgili klinik iddiayı desteklemesi | Yalnız RAG koşullarında incelenen altküme |

Ana ölçümde eksik tahmin ve bozuk çıktı yanlış sayılır; coverage ve yalnız geçerli yanıtlar üzerindeki accuracy ayrıca sunulur. Abstention'ın araştırma anlamı korunur; başarılı sınıflandırma gibi sayılmaz. Sistem arızası ile bilinçli abstention aynı kategoriye konmaz. Altgrup analizi için vaka sayıları ve belirsizlik belirtilir.

Serbest metin eşleştirme önce dondurulmuş klinik eşanlamlı/ontoloji tablosuyla yapılır. Belirsiz eşleşmeler sistem kimliği gizlenmiş klinik adjudication'a gider. LLM judge kullanılacaksa model/sürüm/istem sabitlenir, altkümede insan uyumu ölçülür; tek başına doğruluk hakemi yapılmaz. “melanoma” ve “melanoma in situ” gibi farklı ayrıntı düzeyleri için kural önceden yazılır.

## Bütçe ve yeniden üretilebilirlik

İki sonuç paneli önerilir: doğal yöntem bütçesindeki doğruluk/maliyet ve yaklaşık eşit kaynak bütçesindeki doğruluk. Aynı max_output_tokens değerini her çağrıya vermek, bir ve yedi çağrılı yöntemleri eşit bütçeli yapmaz. Bütçe bir vaka boyunca tüm çağrıları kapsar. Kesin token eşleştirme mümkün değilse gerçekleşen kullanım gösterilir ve “eşit bütçe” ifadesi daraltılır.

Her koşul için model ID, temperature, seed desteği, tekrar numarası, istem hash'i, corpus hash'i, dataset hash'i, commit SHA, API tarihi, timeout, retry ve yanıt şeması kaydedilir. Desteklenmeyen seed sahte bir determinism garantisi gibi yazılmaz. Üç tekrar başlangıç önerisidir; sıcaklık sıfır olsa bile API deterministik varsayılmaz.

Örnek iş yükü: 200 test vakası × 6 koşul × 2 model × 3 tekrar = 7.200 vaka-koşul çalışması. Ortalama dört model çağrısı varsayılırsa 28.800 çağrı eder. Bu yalnız planlama hesabıdır; ücret teklifine veya gerçek maliyet ölçümüne karşılık gelmez. Pilot, gerçek çağrı/token dağılımını verince bütçe hesaplanır.

Başka modele otomatik fallback kapatılır. Retry'lar ölçülen maliyete dahil edilir. Kesilen/başarısız koşular da sonuca kaydedilir. Resume anahtarı vaka+koşul+model+tekrar+konfigürasyon hash'i olmalı; yalnız case_id yeterli değildir. Önbellek aynı deney kimliğine bağlı olmalı, farklı yöntemlerin yanıtları karışmamalı.

## İstatistik ve hata analizi

Aynı vakalarda yöntem farkı için eşleştirilmiş bootstrap güven aralıkları; aynı kaynakta çok vaka varsa kaynak düzeyinde cluster bootstrap önerilir. Model tekrarları bağımsız yeni hasta sayılmaz; vaka başına tekrarları koruyan hiyerarşik/eşleştirilmiş analiz gerekir. Önceden seçilmiş iki yöntemin ikili doğruluk farkında McNemar testi yardımcı olabilir. Çok sayıda karşılaştırma yapılırsa düzeltme ve keşifsel analiz ayrımı yazılır.

Öncelikli karşılaştırmalar A1–A0, A2–A1, A4–A3 ve A5–A2 olabilir. A4–A3 karşılaştırmasında gerçekleşen kaynak farkı gösterilmeden tartışmaya nedensel üstünlük atfedilmez. Etki büyüklüğü ve güven aralığı, yalnız p değerinden önce raporlanır.

Kör hata incelemesinde eksik bilgi, yanlış tanı eşleştirmesi, retrieval hatası, yanlış kanıta dayanma, tartışma sonrası doğru cevaptan vazgeçme, aşırı güven, geçersiz yanıt ve maliyet taşması ayrılır. Başarılı ve başarısız örnekler önceden tanımlı örnekleme kuralıyla seçilir; yalnız iyi örnekler gösterilmez.

## Makale ve kabul ölçütleri

Makale sırası: problem ve 2026 literatür boşluğu; veri ve harness tanımları; ana doğruluk/maliyet tablosu; model × yöntem etkileşimi; güven/hata analizi; ablation; sınırlılıklar. Özetin Results bölümü gerçek deney bitmeden sayısal başarıyla doldurulmaz.

Deney hazır kabulü için şu maddelerin tümü gerekir: gerçek tek çağrı baseline; aynı görünür vaka girdisi; test ve RAG ayrımı; klinik inceleme kaydı; tartışma sonrası skorlanan kararın güncellenmesi; sabit model ve gerçek usage; tüm vakaları kapsayan skor paydası; dondurulmuş prompt/config; yeniden çalıştırma yönergesi. Bu ilk yamada yalnız mock ayrımı ve veri preflight başlangıcı tamamlandı.
