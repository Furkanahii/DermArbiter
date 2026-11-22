# DermArbiter: genişletilmiş 2026 literatürü ve özgünlük değerlendirmesi

**Tarama kesiti: 21 Kasım 2026.** İlk rapordaki 14 çalışmaya 14 ayrı çalışma eklenmiştir: toplam 28 odak kayıt. Bu dosya `literature_2026_tr.md` ile birlikte okunmalıdır. Bu sayı, bütün sonuçların tam metin değerlendirildiği veya 2026 literatürünün tüketildiği anlamına gelmez. Mevcut çalışma, kaynakları doğrulanan bir kapsam taramasıdır; sistematik derleme/PRISMA iddiası yoktur.

## 1. Araştırma kararımız

Hocanın üç önceliği korunuyor: **metin girdisi, yaklaşım karşılaştırması ve raporlanabilir deney**. Bir temel modelin çevresine kurulan kontrol akışını burada “harness” olarak adlandırıyoruz. Yeni model eğitmek, görsel temel model eklemek veya çok büyük bir tıbbi bilgi grafiği kurmak ilk makalenin gereği değildir.

İlk incelemeden sonra önemli düzeltme: Dermatoloji ajan literatürü yalnız DermAgent'tan ibaret değil. SkinGPT-X ve SkinMind doğrudan yakın çalışmalar. Ayrıca dermatolojide metin-only değerlendirme de mevcut. Dolayısıyla “ilk dermatoloji ajanı”, “metinle değerlendiren ilk çalışma”, “ilk kanıta dayalı tartışma” veya “ilk adaptif uzman seçimi” iddiaları savunulamaz. Aşağıdaki karşılaştırmadan çıkardığımız öneri, iyi tanımlanmış dermatoloji metin görevlerinde **hangi düzenin, hangi maliyetle ve hangi hata biçimleriyle çalıştığını** ölçmektir.

## 2. Tarama ve doğrulama yöntemi

Üç arama hattı kullanıldı: (i) general agent/harness/evaluation/budget/debate, (ii) medical multi-agent/clinical reasoning/uncertainty/self-correction, (iii) dermatology agent/text-only/benchmark/clinical cases. Ardından bulunan çalışmaların başlıkları, yazar sayfaları ve referans bağlantıları üzerinden yakın çalışma taraması yapıldı. Arama motoru bulguları arXiv, yayıncı, PubMed, ACL Anthology, CVF ve yazarların kendi depolarına bağlandı. Üçüncü taraf özetleri yöntem iddiasına kaynak yapılmadı.

Yeni kullanılan sorgu örnekleri: `2026 medical multi-agent debate uncertainty budget`, `dermatology text-only benchmark 2026`, `dermatology self-correction 2026 agents`, `SkinGPT-X arxiv`, `SkinMind dermatology`, `MedMCQA dermatology subject dataset license`. Verinin 2022 veya daha eski olması, 2026 literatür sınırlamasını ihlal etmez: veri kökeni atıfları ile 2026 ilgili çalışmalar listesi farklı amaç taşır. Hocanın iki eski makalesi de yaklaşım karşılaştırmasına metodolojik örnektir; 2026 taramasına dahil edilmez.

**Okuma düzeyi:** SkinGPT-X, CARA ve When Debate Helps için HTML yöntem metnine erişildi ve ilgili yöntem/deney bölümleri incelendi. Diğer yeni arXiv kayıtları abstract ve resmi metadata düzeyinde tarandı. JAAD metin baskınlığı çalışmasında PubMed kaydı, yazar deposu ve yazarın Mendeley ekleri kullanıldı; yayıncı tam metni okunmuş sayılmadı. DermEVAL için CVF arama indeksindeki resmi abstract/bibliyografik kayıt kullanıldı; sayfa açılışı 403 verdi. ACL kayıtları resmi abstract ve bibliyografik kayıt düzeyindedir. Bu ayrım, kaynaklara atfedebileceğimiz sonuçların sınırını belirler.

**Tarih kuralı:** İlk preprint 2026 veya resmi yayın 2026. Önceki yıl preprint'i olan 2026 yayınları ayrı sürüm olarak işaretlenir ve mükerrer sayılmaz. Bir kabul notu, makalenin resmi proceedings kaydıyla aynı doğrulama seviyesi değildir. ArXiv `2611.05393` sayfası ilk gönderimi 14 Eylül 2026 olarak gösteriyor; kimlik önekiyle uyuşmayan bu metadata kaydı bibliyografya dondurulmadan önce yeniden kontrol edilmelidir. Yılı doğrulandı, ayı kimlikten tahmin edilmedi.

## 3. Genel ajan literatürüne eklenen çalışmalar

| ID / çalışma | Tarih ve durum | Doğrulanan kapsam | Tasarıma etkisi |
|---|---|---|---|
| G5 — Bertalanič ve Fortuna, *The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate* | 29 Nisan 2026 preprint; ACM DOI bağlantısı var | Küçük açık modellerde homojen tartışma, kendi kendini düzeltme ve kontrol koşulları | Birden fazla çağrıya izin verilen tek ajan mutlaka karşılaştırılmalı |
| G6 — Ferreira, Liu ve Zheng, *Beyond Symmetric Agents* | 26 Eylül 2026 preprint; ICLR 2027'ye gönderilmiş, kabul edilmiş sayılmaz | Persona, sıcaklık ve model çeşitliliği; generation-budget eşleştirilmiş oylama kontrolleri | Rol çeşitliliği ile model gücünü karıştırmamak; bağlam taşmasını ve gerçek token tüketimini kaydetmek |
| G7 — Zhao ve ark., *When Debate Helps: Proposal Supply and Verification-Aware Readout in Multi-Agent Reasoning* | 3 Ekim 2026 preprint | Doğru adayın üretilebilmesi ile doğru adayın seçilmesini ayırır; sabit önerilerle müdahaleler kullanır | İyileşme/bozulma ayrıştırması ve aday havuzundaki doğru cevabı ölçmek zaten literatürde var; bunları yeni metrik diye sunmayacağız |

Kaynaklar: [G5](https://arxiv.org/abs/2605.00914), [G6](https://arxiv.org/abs/2609.35875), [G7](https://arxiv.org/abs/2610.04686).

Bu deneyler dermatolojiye ait değildir. Bulguları “tartışma her zaman yararsızdır” diye genelleyemeyiz. Çalışmamızdaki rolleri, güçlü kontrol gruplarını gerekçelendirmektir. G7 özellikle önemlidir: başlangıç oylaması doğruyken son cevabın bozulması veya yanlışken düzelmesi, DermArbiter'in icat ettiği bir analiz olmayacaktır.

## 4. Medikal ajan literatürüne eklenen çalışmalar

| ID / çalışma | Tarih / okuma düzeyi | Yöntem ve görev | DermArbiter ile örtüşme ve sınır |
|---|---|---|---|
| M7 — Wang ve Yang, *The Consistency Illusion: How Multi-Agent Debate Hides Reasoning Misalignment* | 7 Haziran 2026; yöntem metni | MedQA-USMLE ve MedThinkBench üzerinde CARA ile gerekçe uyumu; Grounded Debate Protocol ile olgu/iddia temelli iletişim | “Ajanlar uzlaştıysa doğru akıl yürüttüler” varsayımını sınar. Olgulara bağlı tartışma tek başına yeni değil |
| M8 — Hou ve ark., *Learning When to Collaborate: Selective Multi-Agent Medical Reasoning via Uncertainty-Aware Routing* (SMART-Med) | 24 Ekim 2026; abstract; arXiv NeurIPS kabul notu | Medikal VQA'da belirsizlikle doğrudan cevap/uzman altkümesi arasında öğrenilmiş yönlendirme | Belirsizlik eşiğiyle uzman çağırmayı genel yenilik iddiası yapamayız; multimodal ve eğitimli yöntem, bizim sabit text protokolümüzle eşdeğer değil |
| M9 — Ge ve ark., *ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory* | 27 Mart, v2 27 Mayıs 2026; abstract; KDD kabul/DOI kaydı | Hipotez arama, çalışma belleği ve deneyim belleğiyle klinik karar orkestrasyonu | Bellek ve hipotez arama alanı zaten çalışılıyor; test sırasında vaka cevabıyla belleği güncellemek bizim ana deneyde kapalı olacak |
| M10 — Xia ve ark., *A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support* | 4 Eylül 2026 preprint; abstract | Nadir ve zor olgularda yapılandırılmış rol/tartışma/birleştirme | Uzman rolleri ve moderatör eklemek tek başına özgün katkı değil; doğrudan ilgili tasarım referansı |
| M11 — Riccio ve ark., *Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning* | Resmi sayfada 14 Eylül 2026; metadata notuna bakınız; abstract | Tartışmada saptanan boşluklara göre retrieval veya uzman daveti | “Şüpheci ajan ihtiyaç halinde RAG çağırsın” önerisi mevcut yöntemle güçlü biçimde örtüşür |
| M12 — Wasi ve ark., *Is There a Dr. House in the Machine? Towards Clinical and Diagnostic Reasoning in LLM-based Medical Agents* | Kasım 2026, Findings IJCNLP-AACL; resmi kayıt | Akıl yürütme, mimari, araç, özerklik ve embodiment eksenlerinde survey | Literatürü sınıflandırmak için yararlı; deneysel üstünlük kanıtı değil |

Kaynaklar: [M7](https://arxiv.org/abs/2606.08457), [M8](https://arxiv.org/abs/2610.33544), [M9](https://arxiv.org/abs/2603.26182), [M10](https://arxiv.org/abs/2609.05069), [M11](https://arxiv.org/abs/2611.05393), [M12](https://aclanthology.org/2026.findings-aacl.82/).

Medikal çalışmalar arasında görev türü çok önemlidir. Çoktan seçmeli bilgi sorusu, açık uçlu tanı, etkileşimli hasta görüşmesi ve görüntülü VQA aynı sonuç tablosunda tek doğruluk oranı olarak birleştirilemez. Farklı veri, backbone ve bütçelerde bildirilen yüzdeler bizim sistemle doğrudan başarı karşılaştırması değildir.

## 5. Dermatoloji literatürüne eklenen çalışmalar

### D5 — Text-dominant decision-making

Nouyed ve ark., *Text-dominant decision-making by large multimodal models in dermatology clinical challenges: Comment on “AI-assisted dermatologic diagnosis using a large language model”*. JAAD, çevrimiçi 24 Haziran 2026; Ekim sayısı, 95(4):e403–e404. Makale türü mektup/yorumdur. Yazar deposu, 44 JDCR vakasına ait 124 soruyu metin, görüntü ve birleşik koşullarda değerlendirdiğini bildiriyor. Bu, metin-only dermatoloji değerlendirmesi için doğrudan öncüldür. Çoklu soruların aynı vakadan geldiği için bağımsız örnek sayısı gibi kullanılamayacağını da hatırlatır. Bizdeki ajan mimarisi karşılaştırması başka bir sorudur. Yazarların “anlamlı fark bulunmadı” bulgusu, istatistiksel eşdeğerlik kanıtı olarak yeniden yazılmamalıdır. [PubMed](https://pubmed.ncbi.nlm.nih.gov/42341935/), [yazar deposu](https://github.com/iqbalnaved/text-dominant-multimodal-dermatology-eval), [ekler, CC BY 4.0](https://data.mendeley.com/datasets/tmd5kd2m2h/3).

### D6 — DermEVAL

Zhao ve ark., *DermEVAL: A Dermatologist-Reviewed Benchmark for Multimodal Large Language Models*, WACV Mart 2026, ss. 927–937. Resmi abstract, 16 hastalık ve 11.347 görüntü üzerinden VQA ve rapor üretimi bildiriyor. Görsel görev olduğundan görüntüyü kaldırıp aynı gold'u kullanmak otomatik olarak geçerli bir metin benchmark'ı oluşturmaz. Ayrıca DermEVAL, DermoBench, DermaBench ve DermBench farklı kaynaklardır; benzer adlar nedeniyle karıştırılmamalıdır. [CVF kaydı](https://openaccess.thecvf.com/content/WACV2026/html/Zhao_DermEVAL_A_Dermatologist-Reviewed_Benchmark_for_Multimodal_Large_Language_Models_WACV_2026_paper.html).

### D7 — Beyond Diagnosis

Yoo, Kim ve Rosen, *Beyond Diagnosis: Evaluating Recommendation Stability under Counterfactual Skin-Tone Perturbations*, Findings IJCNLP-AACL, Kasım 2026. Resmi abstract, 60 klinik vakadan türeyen 240 hekim doğrulamalı görsel varyantla tanı ve öneri kararlılığını inceliyor. Bizim text-only ilk aşamamızın parçası değildir; ileride demografik karşıolgusal deney yapılacaksa, değişikliklerin tanı ve şiddeti koruduğunu klinisyenlerin doğrulaması gerektiğine somut örnektir. Bu çalışmadaki görsel adalet iddiaları mevcut DermArbiter metin deneyine taşınamaz. [ACL](https://aclanthology.org/2026.findings-aacl.230/).

### D8 — SkinGPT-X

Chen ve ark., *SkinGPT-X: A Self-Evolving Collaborative Multi-Agent System for Transparent and Trustworthy Dermatological Diagnosis*, 27 Mart 2026 preprint. Görsel ve tanısal ajanlar, bilgi getirme, vaka inceleme ve gelişen bellek birleştiriliyor. DDI dahil görsel veri kümeleri, ince sınıflandırma ve nadir hastalık değerlendirmeleri kullanılıyor. Bu nedenle “dermatolojiye özel çok ajanlı sistem + RAG + bellek” birleşimi özgünlük için yeterli değil. Metin deneyinde görsel araçları çıkarılmış bir uyarlama, özgün SkinGPT-X'in reprodüksiyonu diye adlandırılmamalı. [Abstract](https://arxiv.org/abs/2603.26122), [yöntem](https://arxiv.org/html/2603.26122v1).

### D9 — SkinMind

Hao ve ark., *SkinMind: Equipping Agentic RAG with Human-Inspired Graph Probing for Reliable Dermatology AI*, 6 Kasım 2026 preprint. Orkestratörün bir bilgi grafiğinde farklı ayrıntı düzeylerinde kanıt aramasını sağlayan sekiz araç ve multimodal kaynaklar kullanılıyor; altı dermatoloji VQA benchmark'ında değerlendiriliyor. Bu, “dermatolojide adaptif kanıt arama” fikrine çok yakın ve güncel bir öncüldür. Büyük bilgi grafiğini yıl sonuna kadar yeniden kurmayı ana hedef yapmak yerine, sabit ve lisansı açık metin kaynaklarıyla kontrollü karşılaştırma daha uygun. [Kaynak](https://arxiv.org/abs/2611.08935).

## 6. Dahil etmediğimiz veya beklettiğimiz adaylar

* *Towards Trustworthy Dermatology MLLMs* (arXiv:2511.09195): ilk preprint 2025. Yazarın kurumsal sayfası ECCV 2026 kabulü bildiriyor; bu turda resmi proceedings sürümü eşleştirilmedi. 28 odak kayda dahil edilmedi; 2026 yayın sürümü doğrulanınca tek çalışma olarak eklenebilir. [Preprint](https://arxiv.org/abs/2511.09195), [yazar laboratuvarı](https://www.joshuachou.ink/publications/).
* MeDxAgent: yazarın 2026 isimli PDF bağlantısı bulundu; tarih/venue düzeyi kesinleştirilmeden doğrulanmış yayın gibi sayılmadı.
* Genel derlemelerde bulunan retinal/ultrason ajanları: yöntemsel yakınlık sınırlı olduğu için dermatoloji bölümüne taşınmadı.
* MedMCQA/MedQA veri kaynakları: 2026 yeni yöntem sayımına dahil değil; veri kökeni ve yeniden kullanım için ayrıca inceleniyor.

## 7. Özgünlük matrisi: neyi savunabiliriz?

| Olası iddia | En yakın öncül | Karar |
|---|---|---|
| Dermatolojiye çok ajan ekledik | DermAgent, SkinGPT-X | Özgünlük iddiası olarak yetersiz |
| Belirsizlikte uzman çağırdık | SMART-Med ve daha eski adaptif sistemler | Tek başına yeni değil |
| Eleştirmen tartışma sırasında RAG çağırıyor | M11, SkinMind | Güçlü örtüşme; yeni diye sunulmaz |
| Gerekçeleri olgulara bağladık | CARA/GDP, DermAgent | Tek başına yeni değil |
| Metin-only dermatoloji değerlendirdik | D5 | Modalite tek başına yeni değil |
| Doğru cevabın bozulmasını ölçtük | G7 ve tartışma analizleri | Metrik yeniliği değil; uygulama/analiz katkısı olabilir |
| Aynı model ve kontrollü bütçeyle dermatoloji harness'lerini, vaka grupları ve hata geçişleriyle karşılaştırdık | MedMASLab + G5–G7 + D5 en yakın birleşim | Savunulabilir çalışma adayı; kapsamın benzersizliği henüz kanıtlanmış değil |

**Önerilen katkı:** (1) tekrar üretilebilir bir metin değerlendirme hattı, (2) güçlü tek-ajan ve bağımsız örnekleme kontrolleriyle karşılaştırma, (3) başlangıç–son karar geçişleri, maliyet ve görev türü üzerinden dermatolojiye özgü deneysel bulgular. Bunlar sonuç çıkmadan “iyileşme” olarak yazılmamalıdır. İyi tasarlanmış bir negatif sonuç, abartılı bir model üstünlüğü iddiasından bilimsel olarak daha sağlam olabilir.

**İsteğe bağlı yöntem adayı:** cevap değiştirme sırasında vaka metnindeki destekleyici/çürüten bulguları açıkça referanslayan bir revizyon kuralı. Bu da GDP'ye yakın olduğundan yeniliği kabul edilmiş değildir. Yalnız açık bir farklılık tanımlanır, klinik olgu etiketleri güvenilir biçimde oluşturulur ve basit kontrolü geçerse eklenir. Ana makale bu fikrin başarılı olmasına bağımlı olmayacak.

## 8. Literatürden deneye geçiş için yapılacaklar

1. Yakın 6–8 çalışmanın tam metinlerinden model, veri, kontrol grubu, çağrı/token bütçesi, finalizer, test izolasyonu ve kod erişimi alanlarını ikinci bir okuma ile çıkarıp çapraz kontrol etmek.
2. JDCR/DermCase adaylarında metin–cevap ayrımı ve görüntüsüz yeterlilik denetimini tamamlamak; sınav sorularını ayrı bir track olarak tutmak.
3. Pilot sonrasında ana karşılaştırmaları ve bütçe çizelgesini test sonuçlarını görmeden dondurmak.
4. Makale gönderiminden önce 22 Kasım–gönderim tarihi arasını aynı sorgu ailesiyle güncellemek; yeni çalışma çıkarsa related work ve iddiayı değiştirmek.

Bu tur araştırma yönünü belirlemeye yeterli kanıt sağlıyor; son bibliyografya ve bütün çalışmaların çift okuma doğrulaması tamamlanmış değildir.
