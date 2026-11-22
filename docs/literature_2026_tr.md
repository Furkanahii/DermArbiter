# DermArbiter için 2026 literatür raporu

Tarama kesiti: 1 Ocak–21 Kasım 2026. Bu rapor, genel ajan/harness, medikal ajan ve dermatoloji ajanları ile yakın değerlendirme çalışmalarından seçilmiş 14 kaydı kapsayan odaklı bir ilk taramadır. Sistematik derleme veya bütün 2026 yayınlarının eksiksiz listesi değildir. Makale gönderiminden önce yıl sonuna kadar çıkan çalışmalarla güncellenmelidir.

En belirgin yöntemsel sonuç, ajan sayısı veya araç eklemenin tek başına katkı olmadığıdır. DermArbiter için savunulabilir araştırma yönü; aynı temel model, aynı vaka bilgisi, açık maliyet kaydı ve bağımsız test üzerinde yaklaşım karşılaştırmasıdır. Bu değerlendirme aşağıdaki çalışmaların ortak mesajından çıkardığımız tasarım önerisidir; DermArbiter için kanıtlanmış sonuç değildir.

## Dahil etme ve kaynak doğrulama kuralı

İlk arXiv gönderimi 2026 olan preprint'ler ve resmi yayın tarihi 2026 olan yayınlar dahil edildi. Daha eski preprint'i olan 2026 dergi makalesi, yeni bir 2026 fikri olarak sunulmaz; yayın sürümü olarak işaretlenir. arXiv güncelleme tarihi ile ilk gönderim tarihi ayrı değerlendirilir. Aynı çalışmanın preprint ve yayın sürümü iki bağımsız çalışma sayılmaz.

Başlık, yıl ve kapsam arXiv, yayıncı, PubMed veya PMC kayıtlarından kontrol edildi. DermAgent, MedMASLab ve DermCase için yöntem metnine de bakıldı. Diğer kayıtların özeti, erişilen abstract ve bibliyografik kayıt düzeyindedir; tabloda görünmeyen ayrıntılar tahmin edilmedi. Yayıncı sayfası erişilemediğinde PubMed/PMC kullanıldı. Arama sonuçlarındaki üçüncü taraf özetler nihai bilimsel iddia kaynağı yapılmadı.

Kullanılan başlıca sorgular: `2026 agent harness evaluation`, `2026 medical multi-agent benchmark`, `2026 dermatology agent`, `dermatology text-only clinical cases benchmark`, `medical agent cost single agent evaluation`. Tarama arXiv, Nature/npj, MICCAI, PubMed/PMC ve JMIR odaklı yürütüldü. Tarama sonrası tekrarlar çalışma düzeyinde birleştirildi. PRISMA tarama sayıları üretilmedi.

## Genel ajan ve harness çalışmaları

| Kayıt | Tarih ve durum | Doğrulanan katkı | DermArbiter açısından anlamı |
|---|---|---|---|
| G1 — Lou ve ark., AutoHarness | İlk gönderim 10 Şubat 2026; arXiv preprint | Oyun ortamı geri bildirimiyle model etrafındaki kod harness'ini üretme ve düzeltme | Model ağırlığı dışında kontrol akışının da araştırma nesnesi olabileceğini gösterir; tıbbi yarar doğrudan çıkarılamaz |
| G2 — Wang ve ark., Rethinking the Evaluation of Harness Evolution for Agents | İlk gönderim 14 Temmuz, v4 5 Ekim 2026; preprint | Eşitlenmiş geri bildirim/çıkarım bütçeleri ve held-out değerlendirme üzerinden harness gelişimini yeniden sorgular | Prompt/harness ayarı ile test ayrılmalı; güçlü ve bütçesi karşılaştırılabilir baselinelar gerekli |
| G3 — Chen ve ark., AgentCompass | İlk gönderim 15 Temmuz, v3 20 Temmuz 2026; preprint | Benchmark, harness ve environment ayrımı; çalışma izlerini incelemeye uygun altyapı | Veri yükleme, yöntem uygulama ve scoring ayrı arayüzler olmalı |
| G4 — Zhang ve ark., When Agents Fail: Model or Harness—Who's to Blame? | 23 Ekim 2026; “under review” preprint | Model–harness etkileşimini, valid completion ve token yükü dahil ölçütlerle değerlendirir | Bir harness'in tek modeldeki sonucu genellenmemeli; doğru cevap ile tamamlanma ayrı raporlanmalı |

Kaynaklar: [G1 AutoHarness](https://arxiv.org/abs/2603.03329), [G2 Rethinking Harness Evaluation](https://arxiv.org/abs/2607.12227), [G3 AgentCompass](https://arxiv.org/abs/2607.13705), [G4 When Agents Fail](https://arxiv.org/abs/2610.32483).

Bu dört çalışma klinik doğrulama sağlamaz. Bizim çıkarımımız, makalenin birim analizini “model adı” yerine “model × yöntem × veri koşulu” olarak kurmaktır. Harness'in sabit olması ile otomatik evrilmesi de farklı deneylerdir; DermArbiter ilk sürümünde otomatik harness evolution iddiasına ihtiyaç yoktur.

## Medikal ajan çalışmaları

### M1 MedMASLab

Qian ve ark., **MedMASLab: A Unified Orchestration Framework for Benchmarking Multimodal Medical Multi-Agent Systems**. İlk gönderim 10 Mart, incelenen v2 18 Mart 2026; arXiv preprint.

Çalışma farklı çok ajanlı mimarileri ortak protokolle karşılaştıran bir platform sunuyor; abstract 11 mimari, 24 modalite ve 11 klinik benchmark bildiriyor. Yöntem metni çağrı bazında token ve süre muhasebesini açıklıyor. Agent/rol sayısının artırılmasının her görevde daha iyi sonuç vermediğini tartışıyor. DermArbiter için en yakın yöntemsel referanslardan biri; multimodal kapsamı, metin odaklı yeni deneyle aynı görev olduğu anlamına gelmez. [Abstract](https://arxiv.org/abs/2603.09909), [yöntem ve maliyet analizi](https://arxiv.org/html/2603.09909v2).

### M2 Medical MCQA kalibrasyon çalışması

Martinez, **Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA**. İlk gönderim 25 Mart, v2 5 Haziran 2026; hakem sürecinde olduğu belirtilen preprint.

Uzman ajan yanıtlarını tutarlılık kontrolü ve ağırlıklı birleştirme ile değerlendiriyor. Deneyler MedQA ve MedMCQA'nın yüksek anlaşmazlıklı seçilmiş altkümelerine ait; genel klinik dağılıma doğrudan taşınmamalı. Kalibrasyon ve doğruluğun ayrı incelenmesini destekliyor. DermArbiter için önerimiz, anlaşma oranını güven yerine koymadan karar güveninin kalibrasyonunu ayrıca ölçmek. [Kaynak](https://arxiv.org/abs/2603.24481).

### M3 Klinik karar görevlerinde ajan karşılaştırması

Liu ve ark., **Benchmarking large language model-based agent systems for clinical decision tasks**. 18 Şubat 2026, *npj Digital Medicine* 9:259; yayınlanmış makale. DOI: 10.1038/s41746-026-02443-6.

OpenManus ve Manus sistemlerini AgentClinic, MedAgentsBench ve HLE üzerinden değerlendiriyor. Abstract, incelenen koşullarda mütevazı başarı kazançlarına karşı yüksek token/süre yükü bildiriyor. Bu sonuçlar başka ajanlara otomatik genellenemez. Ancak maliyeti ana sonuç tablosuna koymak ve güçlü tek-model karşılaştırmasını korumak için doğrudan ilgili bir örnek. [PubMed kaydı ve abstract](https://pubmed.ncbi.nlm.nih.gov/41708802/).

### M4 DeepRare

Zhao ve ark., **An agentic system for rare disease diagnosis with traceable reasoning**. 18 Şubat 2026, *Nature* 651:775–784; yayınlanmış makale. Önceki preprint 2025 tarihli; burada dahil edilen kayıt 2026 dergi sürümüdür.

Serbest metin fenotipler, HPO terimleri ve genetik verileri araç destekli, kanıt izlenebilir bir tanı akışında birleştiriyor. Çoklu veri kümeleri ve uzman incelemesi kullanıyor. Dermatoloji benchmark'ı değil; DermArbiter için benzerlik kaynak izlenebilirliği ve uzman doğrulamasında. Kaynak sayısı ya da ajan sayısı, bu çalışmanın klinik geçerlilik düzeyini bizim prototipe aktarmıyor. [Yayın metni ve tarih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12999473/), [eski preprint](https://arxiv.org/abs/2506.20430).

### M5 AgentClinic

Schmidgall ve ark., **AgentClinic: a multimodal benchmark for tool-using clinical AI agents**. 27 Nisan 2026, *npj Digital Medicine* 9:499; yayınlanmış makale. Bu giriş 2026 yayın kaydını temel alır; yöntemin ilk kez 2026'da ortaya çıktığı iddia edilmez.

Eksik bilgi altında hasta etkileşimi ve araç kullanımı içeren klinik simülasyonları değerlendirir. Abstract dokuz uzmanlık alanı ve yedi dil kapsamı bildiriyor. Statik vignette tanısı ile etkileşimli hasta simülasyonunun farklı deney türleri olduğunu hatırlatır. İlk DermArbiter makalesinde yeni hasta simülatörü kurmak yerine statik metin görevini açık adlandırmak daha yönetilebilir. [Yayın kaydı](https://pubmed.ncbi.nlm.nih.gov/42045532/), [yayıncı](https://doi.org/10.1038/s41746-026-02674-7).

### M6 Klinik testing harness çerçevesi

Waisberg ve Guarnieri, **Agentic AI in Medicine: Challenges for Responsible Development and the Case for Clinical Testing Harnesses**. Çevrimiçi 23 Eylül 2026; Kasım 2026, *Annals of Biomedical Engineering* 54:3721–3725. DOI: 10.1007/s10439-026-04381-6.

Çok adımlı ajanlarda yalnız son çıktının değil eylem dizisinin gözlenmesi, hata yayılımı ve kapsam sınırlarının değerlendirilmesini savunan bir çerçeve yazısıdır. Sayısal üstünlük benchmark'ı olarak kullanılmamalı. Hocanın “medical cases için harness” yönlendirmesine kavramsal olarak çok yakındır; deney izleri ve hata türlerinin saklanmasını gerekçelendirir. [PubMed kaydı ve abstract](https://pubmed.ncbi.nlm.nih.gov/42776404/).

## Dermatoloji ajanları ve yakın değerlendirme çalışmaları

Bu bölümdeki dört kaydın hepsi ajan sistemi değildir. DermAgent doğrudan ajan çalışması; DermaBench ve DermCase benchmark/veri çalışmaları; DermoGPT model/veri/evaluation çalışmasıdır. Sınıflandırmayı doğru yapmak, “dermatoloji ajan literatürü”nü yapay biçimde genişletmeyi önler.

### D1 DermAgent

Liu ve ark., **DermAgent: A Self-Reflective Agentic System for Dermatological Image Analysis with Multi-Tool Reasoning and Traceable Decision-Making**. 14 Mayıs 2026 arXiv; kayıt MICCAI 2026 erken kabul notu taşıyor. Konferans sayfasındaki başlık **A Multi-Tool Agentic System for Evidence-Grounded and Traceable Reasoning in Dermatology** biçiminde; aynı çalışma ailesi, iki bağımsız kayıt değil. Konferans sayfası erişim doğrulamasına takıldığı için burada arXiv sürümünün yöntemi esas alındı.

Plan–Execute–Reflect döngüsünde görsel araçlar, vaka/guideline retrieval ve deterministik critic kullanıyor. DermArbiter araç bileşenleriyle belirgin örtüşme var. İlk makaleyi görüntüsüz kurduğumuzda yayımlanan görsel accuracy ile doğrudan başarı kıyası yapamayız. Varsa uyarlama, “DermAgent-inspired text adaptation” olarak açıkça adlandırılmalı ve özgün sistemi yeniden üretmiş gibi sunulmamalı. [arXiv](https://arxiv.org/abs/2605.14403), [yöntem](https://arxiv.org/html/2605.14403v1), [konferans kaydı](https://papers.miccai.org/miccai-2026/0263-Paper0286.html).

### D2 DermaBench

Yilmaz ve ark., **DermaBench: A Clinician-Annotated Benchmark Dataset for Dermatology Visual Question Answering and Reasoning**. 20 Ocak 2026; arXiv preprint. Abdurrahim Yilmaz yazarlar arasında.

DDI üzerinde 656 görüntü, 570 hasta ve 22 ana soruluk klinisyen anotasyon şeması bildiriyor. VQA/morfoloji referansı olması, görselsiz doğal klinik vaka verisi olduğu anlamına gelmez. Metin anotasyonlarından yararlanma düşünülürse kaynak lisansı, tanı sızıntısı ve görselden türeme ayrı incelenmeli. Mevcut depodaki “aynı DDI verisini kullanmak sonuçları doğrudan karşılaştırılabilir kılar” fikri, girdi/görev/protokol eşitliği olmadan yeterli değil. [Kaynak](https://arxiv.org/abs/2601.14084).

### D3 DermCase

Liu ve ark., **Mind the Rarities: Can Rare Skin Diseases Be Reliably Diagnosed via Diagnostic Reasoning?** 19 Mart 2026; arXiv preprint.

DermCase, vaka raporlarından 6.354 olgu ve 26.030 multimodal çift oluşturduğunu; tanı, diferansiyel ve reasoning değerlendirdiğini bildiriyor. Uzun klinik bilgi içermesi nedeniyle metin veri adayıdır, fakat özgün görev multimodaldir. Öykü/cevap/reasoning alanlarının ayrılması ve görüntüsüz yeterliliğin klinik kontrolü yapılmadan kullanıma hazır kabul edilmemeli. Lisans ve veri indirme erişimi bu incelemede tamamlanmış bir veri kabul süreci değildir. [Abstract](https://arxiv.org/abs/2603.18418), [veri hazırlama yöntemi](https://arxiv.org/html/2603.18418v1).

### D4 DermoGPT ve DermoBench

Ru ve ark., **DermoGPT: Open Weights and Open Data for Morphology-Grounded Dermatological Reasoning MLLMs**. 5 Ocak 2026; arXiv preprint.

Morfoloji temelli eğitim verisi, model ve 11 görevli DermoBench değerlendirmesini birlikte sunuyor; morphology, diagnosis, reasoning ve fairness eksenlerini içeriyor. Bu, “dermatolojide çok boyutlu değerlendirme yok” iddiasını zayıflatıyor. DermoBench adı, D2'deki DermaBench ile karıştırılmamalı. Model ağırlıkları/verilerinin bugün erişilebilirliği, abstract'taki gelecek zamanlı paylaşım ifadesinden çıkarılmamalı. [Kaynak](https://arxiv.org/abs/2601.01868).

## Karşılaştırmalı araştırma boşluğu

| Araştırma ihtiyacı | Yakın kaynaklar | DermArbiter için önerilen deney |
|---|---|---|
| Model etkisi ile harness etkisini ayırmak | G2, G4, M1 | En az iki sabit model üzerinde aynı koşul matrisi |
| Tartışma ile ek çıkarım bütçesini ayırmak | G2, M1, M3 | Bağımsız örnekleme/oylama kontrolü ve maliyet paneli |
| Metin kaynaklı dermatolojik tanı | D2, D3 | Görüntüsüz yeterliliği doğrulanmış, sızıntısı denetlenmiş vignette |
| Yanıt doğruluğu ve çalışma güvenilirliği | G3, G4, M6 | Tüm-vaka accuracy, valid completion, retry ve hata türleri |
| İzlenebilir kanıt ve güven | M2, M4, D1 | Karara bağlı güven; RAG altkümesinde atıf desteği incelemesi |

Bu matris bir araştırma hipotezidir; bu kombinasyonun literatürde hiç denenmediğini kanıtlamaz. Gönderim öncesinde özellikle “text-only dermatology multi-agent”, “budget-matched medical debate”, “clinical testing harness” ve DermCase/DermaBench atıf zincirleri ayrıca taranmalıdır.

## Raporun makaleye dönüşmesi

Related Work üç bölümde kurulabilir: genel harness değerlendirmesi; medikal çok ajan karşılaştırmaları; dermatoloji veri ve ajan sistemleri. Her bölümün sonunda mevcut çalışmalara karşı savunulabilir fark tek cümleyle açıklanmalı. Preprint sonuçları hakemli bulgu gibi etiketlenmemeli. Çalışmalar arasında veri, etiket uzayı, modalite veya split farklıysa başarı yüzdeleri ortak sıralama tablosuna konmamalı.

2020/2021 danışman örnekleri yalnız deney tasarımının tarihsel referanslarıdır. 2025 PanDerm, MAKE, Derm1M ve benzeri kullanılan bileşenlerin kaynakları normal makale kaynakçasında gerekebilir; “yalnız 2026 literatür raporu” kısıtı, kullanılan eski veri/modeli kaynak göstermemeyi gerektirmez. Bu iki amaç ayrı tutulmalıdır.

İlk yazılabilir katkı cümlesi: “We compare a controlled set of text-based dermatology reasoning approaches across fixed model backbones, evaluating diagnostic performance, completion reliability, confidence calibration, and inference cost.” Bu cümle çalışma amacıdır; yeni deney sonuçları üretilince kapsamı gerçekleşen tasarıma göre güncellenir.
