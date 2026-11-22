# DermArbiter çalışma incelemesi ve makaleye hazırlık

21 Kasım 2026. İncelenen temel sürüm: `main`, `f63e22e368d6812350750a2ccb0554e86ca6f32a`. Yerel başlangıç dalı: `research/text-only-readiness`.

DermArbiter, yeniden kullanılabilir ajan, araç, orkestrasyon ve değerlendirme bileşenleri olan bir araştırma prototipi. Ancak mevcut varsayılan deney yolları ve ölçüm tanımları, hocanın istediği metin tabanlı karşılaştırmayı henüz güvenilir biçimde üretmiyor. İlk iş daha fazla model eklemek değil; deney koşullarını, veri akışını ve skorların neyi ölçtüğünü düzeltmek. Yaklaşık altı haftada tamamlanmış bir makale taslağı ve gönderim paketi hedeflenebilir; hakem kabulü ve yayın tarihi bu takvime bağlanamaz.

## Hocanın önerilerinin araştırma kararlarına dönüşümü

| Öneri | Çalışmaya etkisi | Somut çıktı |
|---|---|---|
| Mevcut çalışmayı benchmark olarak konumlandırmak | Temel model başarımı ile onu çevreleyen kontrol akışının katkısını ayırmak | Aynı vakalar ve model sürümleri üzerinde yöntem karşılaştırması |
| Yaklaşımlar kümesini kendi içinde karşılaştırmak | Önceden kazanacağı varsayılan bir yöntemden kaçınmak | Doğrudan yanıt, yapılandırılmış yanıt, RAG, örnekleme, tartışma koşulları |
| Text-only çalışmak | Görüntü, görüntü gömmesi ve görsel araç çıktısını ana deneyden çıkarmak | Açık metin veri sözleşmesi ve izinli girdi alanları |
| Görüntüyü ikinci aşamaya bırakmak | Görsel model kurulumu, GPU ve görüntü benchmark tekrarını kritik yoldan çıkarmak | İkinci aşama için korunan mevcut araç modülleri |
| Yalnız 2026 literatürü | Preprint ve hakemli yayın durumunu ayrı belirtmek | Üç kategorili literatür raporu ve tarih kuralı |
| Çalışmanın raporu | Yazılan özellikler, çalışan kod ve deneysel kanıtı ayırmak | Bu inceleme, deney protokolü, eksik işler ve doğrulama kayıtları |

Hocanın [2020 Pap Smear karşılaştırması](https://arxiv.org/abs/2009.06366) ile [2021 ISIC mimari karşılaştırması](https://arxiv.org/abs/2110.12270), aynı problem üzerinde alternatifleri karşılaştırmanın örnekleri. Bu bağlantıları yöntemsel örnek olarak tutmak uygun; 2026 literatür tablosuna dahil etmek uygun değil. Görsel çalışmaların başarı yüzdeleri, farklı girdili yeni deneyin doğrudan referans skoru olamaz.

## İncelemenin kapsamı

GitHub varsayılan dalı klonlandı; açık PR listesi kontrol edildi ve boş bulundu. Diğer dalların varlığı doğrulandı; her dalın ayrı deneyi yeniden çalıştırılmadı. Temel değerlendirme main sürümüne aittir. Core, agents, tools, experiments, evaluation, scripts, configs, tests ve araştırma belgeleri üzerinden ana veri ve sonuç akışları izlendi. Notebookların tamamı gerçek modellerle çalıştırılmadı. API/GPU çıkarımı ve klinik doğruluk deneyi yapılmadı.

Takip edilen dosyalarda beş örnek vaka, 51 boundary probe, 10 guideline seed kaydı, HAM10000/Fitzpatrick17k metadatası ve DermAgent altküme CSV'si var. Klinik incelemesi tamamlanmış birincil metin test kümesi veya gerçek model koşularına ait yayın sonucu dosyası bulunmadı. Bu, ekipte depo dışında sonuç olmadığı anlamına gelmez; o kayıtlar ayrıca alınmalı. Belgelere yazılmış 45 probe ve 428 test gibi sayılar mevcut içerikle yeniden uzlaştırılmalı.

## Yeniden kullanılabilir güçlü taraflar

Pydantic tabanlı blackboard, vaka durumu ile kanıt kartlarının ayrımını sağlıyor. LangGraph akışı fazları görünür kılıyor. Rol sınıfları ve model router yeni kontrollü koşullar için temel oluşturuyor. Mock ortam, API çağrısı olmadan davranış testleri yazmaya elverişli. JSONL sonuçları, ablation altyapısı, etiket normalizasyonu, boundary testleri ve klinisyen çalışma sayfaları sıfırdan yazılması gerekmeyen bileşenler.

Bu altyapı değerli; fakat birim testlerinin geçmesi, metin öyküsünün gerçekten modele ulaştığını veya tartışmanın tanıyı değiştirdiğini kendiliğinden kanıtlamıyor. Kabul testleri araştırma hipotezinin gerektirdiği davranışı kontrol etmeli.

## Kritik bulgular

### Gerçek çalışma seçeneği bazı yollarda mock sonuç üretiyor

`experiments/baseline.py::_build_baseline_orchestrator`, `full_panel.py::_build_full_orchestrator` ve `ablation.py::_run_variant`, özgün sürümde `mock` değerine bakmadan mock factory kullanıyor. Buna karşılık `experiments/runner.py` ve `evaluation/benchmark_runner.py` gerçek yol uygulanmadığında açık hata veriyor. Ayrı `scripts/run_dermagent_subset.py --real` yolu ise gerçek pipeline kurabiliyor. Dolayısıyla sorun bütün depoda gerçek çıkarımın bulunmaması değil; deney giriş noktalarının farklı davranması.

Etkisi: CLI çalıştı diye gerçek model deneyi yapıldığı sanılabilir. Bu ilk yamada üç problemli yola açık hata eklendi; dört deney runner'ının ürettiği mock satırları `execution_mode=mock` ve `publication_eligible=false` taşıyor. Eski sonuçlar bu alanlar yokken otomatik olarak gerçek kabul edilmemeli. Tüm analiz yollarında mock kayıt reddi henüz uygulanmadı.

### Mevcut baseline tek çağrılı ve araçsız değil

`BaselineRunner`, specialist ve moderator rollerini, mock araç havuzunu ve `max_rounds=1` kullanıyor. Bir tur sınırı, sıfır tartışma anlamına gelmez; rakip brief olmadığı için bazı turlar atlanabilir. Baseline adı araştırma koşulunu doğru anlatmıyor. Üstelik docstring'deki “lower-bound” ifadesi deneyden önce bir sıralama varsayıyor.

Gerekli düzeltme: Sıfır araç, sıfır moderator, tek model çağrısı yapan yeni bir yol. Tüm koşullar aynı cevap şemasını kullanmalı. Eski yol uyumluluk için korundu ve açıklaması düzeltildi; gerçek baseline bu yamada tamamlanmış değil.

### Klinik öykü bağımsız değerlendirme promptuna doğrudan aktarılmıyor

`core/debate_protocol.py:109` içindeki `independent_read`, ajanlara yalnız `state.evidence_cards` veriyor. `agents/specialist.py:81` bu kartlardan prompt oluşturuyor. `state.query` ve `patient_context` doğrudan brief girdisi değil. Araçlar bu bilgiyi geri döndürürse dolaylı bağlam oluşabilir; bu, araçsız metin koşulu için yeterli değil. DermArena satırındaki ayrı `clinical_history` alanı da eski subset adapter'da state'e taşınmıyor.

Yerel kontrollü yeniden üretimde `UNIQUE_CASE_HISTORY_TOKEN` query alanına kondu; specialist'e gönderilen promptta bulunmadı. Gerekli davranış: Her koşul aynı görünür vaka metnini almalı; kaynak bilgisi ve gold tanı yalnız değerlendiricide kalmalı. Yeni preflight'ın `model_input()` işlevi bu ayrım için başlangıçtır; eski runner'lara henüz bağlanmamıştır.

### Tartışma ölçülen tanı sıralamasını güncellemiyor

`targeted_debate` argümanları debate log'a ekliyor, fakat `state.briefs` içindeki tanı/güven değerlerini revize etmiyor. `synthesis` final tanı sıralamasını bu eski brief'lerden hesaplıyor; moderator'ın son raporu ayrı bir metin. Sonuç olarak anlatı değişebilirken skorlanan tanı eski oy birleştirmesinden kalıyor.

Kontrollü testte dört tartışma mesajı açıkça başka tanı önerdi; önceki ve sonraki `final_diagnosis` aynı kaldı. Bu gözlem gerçek klinik başarımı ölçmez, veri akışı kusurunu gösterir. Gerekli düzeltme: Her tur sonunda yapılandırılmış yeniden değerlendirme veya tartışmayı okuyan yapılandırılmış son hakem kararı. Skorlanan tanı ve gösterilen rapor aynı son karardan türemeli. Tartışmasız ensemble ayrı koşul olmalı.

### Eksik tahminler değerlendirmeden düşebiliyor

`evaluation/dermarena.py:164`, gold ve prediction kayıtlarını case_id üzerinden inner join ile eşliyor. Tekrarlanan kimlikler dict oluşturulurken önceki kayıtları örtebilir. İki gold vaka ve yalnız bir doğru tahmin içeren örnekte `n_cases=1`, accuracy=1.0 elde edildi. Sabit iki vakalık test kümesi açısından tamamlanma %50 ve tüm-vaka doğruluğu %50 olmalıydı.

Gerekli düzeltme: Gold manifest değerlendirme evrenini belirlemeli. Eksik kayıt, timeout, parse error ve abstention ayrı durumlar olarak korunmalı. Ana doğruluk tüm test vakaları üzerinden, tamamlananlar üzerindeki doğruluk ayrıca raporlanmalı. Duplicate ve beklenmeyen kimlikler hata vermeli. Bu skorlayıcı henüz değiştirilmedi.

### Anlaşma oranı güven olasılığı olarak kullanılıyor

`synthesis` consensus değerini en sık birinci tanıya katılan ajan oranı olarak hesaplıyor. `dermarena.py:327` ve `experiments/analyze.py` bu değeri ECE/Brier için kullanıyor. Tek oy veren ajanlı durumda oran her zaman 1 olur; üç ajan için kaba ayrık değerler oluşur. Aynı fikirde olmak, doğru olma olasılığıyla aynı değişken değildir.

Consensus ayrı bir süreç ölçütü olmalı. Kalibrasyon için son top-1 kararına bağlı, tanımı bütün koşullarda aynı olan güven değeri gerekir. Yalnız top-1 güvenle hesaplanan Brier, çok sınıflı olasılık vektörü Brier skoru diye sunulmamalı. Post-hoc calibration uygulanırsa parametreler yalnız development verisinde öğrenilmeli.

### Görüntü alanını kapatmak yeterli değil

`tools/case_rag.py` görüntü gömmesi hesaplıyor ve görüntüsüz girdiyi reddediyor. FairnessProbe görüntüden ITA/renk temelli bir tahmin çıkarıyor. PanDerm, MAKE ve görsel VQA araçları birinci aşamanın girdisine uygun değil. Guideline RAG metinle kullanılabilir; fakat corpus, chunking, embedding sürümü, kaynak listesi ve erişim tarihi sabitlenmeli.

Ana text-only koşulunda görüntüler, görsel model çıktıları ve tanıdan türetilmiş açıklamalar bulunmamalı. Görüntüden yazılan insan betimleri veya caption'lar değerlendirilecekse bunlar ayrıca “betim üzerinden akıl yürütme” koşulu olarak etiketlenmeli; doğal klinik öykü ile birleştirilmemeli.

### Araç planlama ve bütçe kontrolünün anlamı sınırlı

Specialist'in `propose_tools` uygulaması sabit `_DEFAULT_TOOLS` listesini döndürüyor. Bu davranış dinamik araç seçimi olarak adlandırılmamalı. Debate token sayımı çoğunlukla karakter tabanlı kestirim; bazı argümanlar modelden üretildikten sonra kesiliyor. Böyle bir kesme, API'de harcanan tokenı geri almaz. Global bütçe de bütün fazlardaki gerçek kullanımı kesin biçimde sınırlandırmıyor.

Gerekli kayıt: Her çağrının input/output tokenı, gerçek model kimliği, retry, süre, hata ve tool çağrısı. Token sayacı tahminse alan adı bunu söylemeli. Para hesabı model bazlı giriş/çıkış fiyatları ve fiyat tarihiyle yapılmalı; tek blended sabit fiyat heterojen panel için yeterli değil.

### Model fallback karşılaştırma koşulunu değiştirebiliyor

`core/model_router.py:158` bir backend başarısız olduğunda başka backend ve model adına geçebiliyor. Ürün dayanıklılığı açısından yararlı olabilen bu davranış, “model A üzerinde yöntem X” deneyini karıştırır. Benchmark modunda model değiştiren fallback kapalı olmalı; aynı modelde sınırlı retry ayrıca kaydedilmeli. Fallback başlı başına incelenecekse ayrı deney koşulu olmalı.

### Bazı kalite ölçütleri vekil ölçüt

`grounding()` yalnız citation kartı bulunup bulunmadığını sayıyor. Kart olması, kaynağın iddiayı desteklediğini göstermiyor. Araçsız koşula otomatik sıfır verilmesi, tüm koşulları kapsayan bileşik skoru RAG lehine yapısal olarak kaydırıyor. Citation varlığı, kaynak doğruluğu ve iddia desteği ayrı ölçülmeli.

`state_to_dermarena_prediction` yönetim/triyaj bilgisini tanı sınıfından türetiyor. Bu, modelin gerçekten söylediği triyajı ölçmüyor. Eşleme gold üretiminde de kullanıldığında bağımsız karar kalitesi sanılan bir döngü oluşabilir. Klinik olarak incelenmiş referans ve modelin açık yapılandırılmış kararı gerekli.

Dokuz boyutlu composite, uygulanabilir boyutlar arasında ortalama alıyor. Farklı eksik boyutlar iki sistemin farklı paydalarda karşılaştırılmasına yol açabilir. Ana tabloda ayrı metrikler kullanılmalı; composite birincil sonuç olmamalı.

### Fairness iddiaları mevcut kanıttan güçlü

README'deki “fairness guarantees”, “top-tier performance” ve kalibre belirsizlik ifadeleri, bu depoda izlenebilir deney kanıtıyla desteklenmiyor. Görüntüden renk analizi veya subgroup accuracy hesaplayıcı bulunması adillik garantisi sağlamaz. DDI loader'ındaki 12/34/56 gruplarını II/III/V temsilci tiplere dönüştürmek de 34 grubunu tek tarafa atıyor; özgün grup tanımı korunmalı.

Metin aşamasında demografik veri varsa keşifsel subgroup sonuçları sunulabilir. Grup örneklem sayısı, güven aralığı ve vaka karışımı verilmeden sonuç genellenmemeli. Bu aşamada geniş fairness ve klinik güvenlik iddiaları ana katkıdan çıkarılmalı.

### Veri kaynağı ile gold standard aynı şey değil

DDI yükleyicisinde öykü boş; HAM10000 girdisi çoğunlukla genel soru ve metadata. Derm1M caption'ı doğrudan öykü alanına taşınıyor; caption tanı adını veya patoloji sonucunu içerebilir. Otomatik enrichment klinisyen referans kararının yerine geçmez. Sentetik üretici kendisinin makale sonuçları için olmadığını zaten belirtiyor, buna rağmen annotation_status=frozen verebiliyor; tek bir frozen alanı bu nedenle yeterli değil.

Gerekli denetimler: Kaynak ve lisans kaydı, tanı öncesi görünür bilgi sınırı, kimliksizleştirme, cevap sızıntısı incelemesi, hasta/olgu/yayın düzeyinde gruplanmış ayrım, RAG kaynaklarıyla örtüşme kontrolü ve klinik referans doğrulaması. Bunların yapıldığı henüz varsayılamaz.

## İlk uygulama paketinin kapsamı

1. Üç deney yolunda gerçek çalışma isteğinin sessiz mock üretmesi engellendi.
2. Dört deney runner'ının mock çıktıları açıkça işaretlendi.
3. Legacy baseline açıklaması gerçek yapısına uygun hale getirildi.
4. `scripts/validate_text_benchmark.py` eklendi. Boş/bozuk veri, duplicate case_id, dev/test kaynak ve normalize metin örtüşmesi, görüntü alanı, bekleyen anotasyon, otomatik reviewer, uygunsuz metin kökeni ve izin verilmeyen hasta bağlamı alanlarını denetliyor. Dosyanın SHA-256 değerini kaydediyor.
5. Model-visible alanları seçen yardımcı işlev, gold ve kaynak kimliğini girdiden ayırıyor.
6. 22 yeni regresyon testi eklendi. Gerçek bir model veya klinik veri kullanılmadan hepsi geçti.

Bu bir hazırlık yamasıdır. Text-only gerçek runner, tartışma sonrası karar revizyonu, kalibrasyon düzeltmesi, eksik tahmin muhasebesi ve ölçülmüş maliyet henüz tamamlanmadı. Preflight, beyan edilen lisans ve klinisyen incelemesini bağımsız doğrulamaz; metnin içindeki anlamsal cevap sızıntısını yakaladığını iddia etmez.

## Doğrulama sonuçları ve sınırları

İzole Python 3.12.1 ortamında geniş CPU odaklı test çalıştırması **598 geçti, 4 atlandı** sonucunu verdi. Bu sayıya 22 yeni regresyon testi dahildir. `test_week2_tools.py`, `test_model_router.py` ve `test_panderm.py` PyTorch/Transformers/görsel bağımlılıkları kurulmadığı için bu koşunun dışında bırakıldı; bütün deponun tüm bağımlılıklarla doğrulandığı söylenemez. İlk koşudaki beş hata eksik `huggingface_hub` ve `tqdm` kurulduktan sonra giderildi; bunlar uygulama regresyonu değildi.

Komut: `python -m pytest tests/ --ignore=tests/test_week2_tools.py --ignore=tests/test_model_router.py --ignore=tests/test_panderm.py -q -o addopts= --disable-warnings`. Kurulu sürümler teslim paketindeki `review-environment.txt` içinde kayıtlı; bu dosya üretim dependency lock'u değildir. `git diff --check` ve yeni Python dosyalarının derleme kontrolü de geçti. Beş örnek vaka yeni birincil veri sözleşmesini geçemedi; bu beklenen bir sonuç, veri kaybı veya klinik performans ölçümü değil.

### Bulguların kaynak kod bağlantıları

- [Özgün baseline kurulumu](https://github.com/Furkanahii/DermArbiter/blob/f63e22e368d6812350750a2ccb0554e86ca6f32a/dermarbiter/experiments/baseline.py#L74)
- [Öykü aktarımı ve tartışma akışı](https://github.com/Furkanahii/DermArbiter/blob/f63e22e368d6812350750a2ccb0554e86ca6f32a/dermarbiter/core/debate_protocol.py#L109)
- [Tanı sıralamasının birleştirilmesi](https://github.com/Furkanahii/DermArbiter/blob/f63e22e368d6812350750a2ccb0554e86ca6f32a/dermarbiter/core/debate_protocol.py#L273)
- [Eksik tahminlerin eşleştirilmesi](https://github.com/Furkanahii/DermArbiter/blob/f63e22e368d6812350750a2ccb0554e86ca6f32a/dermarbiter/evaluation/dermarena.py#L164)
- [Kalibrasyon ölçümü](https://github.com/Furkanahii/DermArbiter/blob/f63e22e368d6812350750a2ccb0554e86ca6f32a/dermarbiter/evaluation/dermarena.py#L327)
- [Model fallback akışı](https://github.com/Furkanahii/DermArbiter/blob/f63e22e368d6812350750a2ccb0554e86ca6f32a/dermarbiter/core/model_router.py#L158)

## Makalenin önerilen çerçevesi

Önerilen başlık: **DermArbiter: A Controlled Benchmark of Agentic Harnesses for Text-Based Dermatological Case Reasoning**.

Araştırma sorusu: Aynı temel modeller ve vaka bilgisi altında, yapılandırılmış istem, bilgi getirme, bağımsız örnekleme ve çok ajanlı tartışma; tanı doğruluğunu, tamamlanma oranını, güven kalibrasyonunu ve hesaplama maliyetini nasıl değiştiriyor?

“Improved” sözcüğü olumlu sonuç elde edilmeden başlıkta kesinleştirilmemeli. Sonuç tek modelin daha ucuz ve aynı derecede başarılı olduğunu gösterirse de çalışma değerlidir. Özgünlük önerisi, alan içi kontrollü deney ve yeniden üretilebilir hata/maliyet analizidir; dünyada ilk olma iddiası değildir.

Mevcut Nature Medicine outline'ındaki “üstünlüğü kanıtlama” ve örnek başarı ifadeleri sonuçlarla doldurulmayı bekleyen hipotezler olarak dahi dikkatle yeniden yazılmalı. Dergi seçimi yöntem, veri, klinik katılım ve sonuç düzeyi görüldükten sonra yapılmalı. Yıl sonu için denetlenebilir hedef: tamamlanmış metin, kod, veri/provenance belgesi, deney çıktıları ve gönderim dosyaları.

## 31 Aralık 2026 çalışma takvimi

| Tarih | Hedef | Tamamlanma ölçütü |
|---|---|---|
| 21–24 Kasım | Kapsam ve yöntem | Metin görev tanımı, ana metrik, karşılaştırma matrisi ve model adayları yazılı |
| 25–30 Kasım | Veri ve runner | Gerçek tek çağrı baseline; tüm ajanlarda aynı öykü; incelenmiş pilot; kaynak bazlı ayrım |
| 1–6 Aralık | Pilot ve hata düzeltme | 20–30 development vakası; gerçek usage; hata türleri; tartışmanın karara etkisini gösteren test |
| 7–13 Aralık | Dondurma ve ana deney | Prompt/model/corpus/split hash'leri; kilitli test; tekrar koşuları; eksiksiz kayıt |
| 14–20 Aralık | Analiz ve klinik inceleme | Eşleştirilmiş güven aralıkları; maliyet tablosu; kör hata incelemesi; sonuç bölümü |
| 21–27 Aralık | Tam makale | Introduction, Methods, Results, Discussion; ekler ve yeniden üretim yönergeleri |
| 28–31 Aralık | Son kontrol ve gönderim hazırlığı | Yazar geri bildirimi; kaynak/şekil kontrolü; hedef dergi formatı; preprint/gönderim paketi |

30 Kasım'da kullanılabilir veri ve gerçek baseline yoksa kapsam 4 yöntem × 2 model düzeyine küçültülmeli. 6 Aralık'ta pilot tamamlanmamışsa yeni araç, görev ve veri kaynağı ekleme durmalı. Ana testte görülen sonuçlara göre prompt ayarlanmamalı; değişiklik gerekiyorsa yeni protokol sürümü ve bağımsız test ayrımı gerekir.

## Ekipte netleştirilecek kararlar

Depo dışındaki gerçek sonuçlar ve veri nerede; klinik referansı kim doğrulayacak ve kaç vaka inceleyebilecek; hangi API/GPU erişimi ve toplam bütçe var; yıl sonu hedefi gönderilmiş makale mi yoksa hazır taslak mı? Bunlar yanıtlanana kadar kapsam ve bütçe sayıları öneridir. Hoca teknik danışmansa, dermatolojik gold standardı ayrıca yetkin klinik değerlendirici doğrulamalı; unvandan klinik uzmanlık varsayılmamalı.

Furkan için runner, model girdisi, kullanım kaydı ve yapılandırma; Emre için veri/provenance, skorlayıcı ve istatistik; danışman ve klinik değerlendirici için görev kapsamı, referans cevap ve kör hata incelemesi dağılımı önerilebilir. Bu görev dağılımı henüz ekip kararı değildir.
