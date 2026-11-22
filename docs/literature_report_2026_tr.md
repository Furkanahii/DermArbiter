# 2026 Literatür Raporu — Genel, Medikal ve Dermatoloji Ajanları

**Hazırlanma:** 22 Kasım 2026 · **Kapsam:** 1 Ocak – 22 Kasım 2026 · **Amaç:** *A Use Case of Agentic Harnesses for Medical Cases: A Controlled Text-Only Benchmark in Dermatology* çalışmasını literatürde konumlandırmak.

Bu rapor 21 Kasım'daki iki ön taramayı (`docs/literature_2026_tr.md`, `docs/literature_expanded_2026_tr.md`) birleştirir, 22 Kasım taramasıyla genişletir ve hocanın istediği üç başlığa ayırır.

## Dahil etme kuralı ve doğrulama

* **Yalnız 2026:** İlk arXiv gönderimi 2026 olan preprint'ler veya resmi yayın tarihi 2026 olan makaleler. Daha eski preprint'i olan bir 2026 dergi makalesi "2026 yayın sürümü" diye işaretlendi; yeni fikir gibi sunulmadı. 2025 preprint'i olup 2026 yayın sürümü eşleşmeyenler dışarıda bırakıldı (§5).
* **Doğrulama:** **[A]** = başlık ve ilk gönderim tarihi arXiv API'den kontrol edildi. **[P]** = başlık, dergi, tarih ve DOI PubMed'den kontrol edildi. **[O]** = başlık, yazarlar, venue ve sayılar resmi bildiri sayfasında (ACL Anthology) 22 Kasım'da kontrol edildi. **[W]** = yalnız resmi web sayfasından alındı, yeniden doğrulanamadı. Üçüncü taraf özetler kaynak olarak kullanılmadı.
* Bu rapor sistematik bir derleme değildir (PRISMA sayımı yok). Gönderimden önce Aralık sonunda bir kez daha güncellenmelidir.

**En önemli 3 bulgu**

1. **Harness, medikal ajan değerlendirmesinde artık açıkça bir araştırma nesnesi.** MedicalHarness (Ekim 2026) aynı modeli farklı harness'lerde koşturarak sonuç değişkenliğinin yaklaşık dörtte birinin harness ve harness × model etkileşiminden geldiğini ve tek bir harness'in her yerde en iyi olmadığını bildiriyor. Hocanın önerdiği "harness for medical cases" çerçevesi güncel ve savunulabilir. Ama "harness'i ilk biz inceledik" denemez.
2. **"Çok ajan her zaman daha iyi" fikri 2026'da açıkça sorgulanıyor.** Cost of Consensus, When Debate Helps, Liu ve ark. (npj) ve MedMASLab tartışma ve çok ajanlı yapıların eşit bütçeli tek ajan kontrollerine karşı sınırlı, göreve bağlı ve pahalı kazanç verdiğini gösteriyor. Hocanın "en iyisini yaptık değil, yaklaşımları karşılaştırdık" yönlendirmesiyle birebir örtüşüyor.
3. **Dermatoloji ajanlarının hemen hepsi görüntü merkezli** (DermAgent, SkinGPT-X, SkinMind, DERM-3R). Aynı hasta vakalarında *metin-only* ve kontrollü bir harness karşılaştırması bulamadık. Bu bizim boşluğumuz.

---

## 1. Genel ajan ve harness çalışmaları

| # | Çalışma | Tarih / durum | Ne yapıyor | Bizim için anlamı |
|---|---|---|---|---|
| G1 | Lou ve ark., **AutoHarness**: improving LLM agents by automatically synthesizing a code harness [A] [2603.03329](https://arxiv.org/abs/2603.03329) | 10 Şub 2026, preprint | Model etrafındaki kod harness'ini otomatik üretip iyileştiriyor | Harness'in modelden ayrı bir tasarım nesnesi olduğunu gösteriyor; otomatik harness evrimi bizim kapsamımızda değil |
| G2 | Wang ve ark., **Rethinking the Evaluation of Harness Evolution for Agents** [A] [2607.12227](https://arxiv.org/abs/2607.12227) | 14 Tem 2026 (v4 Eki) | Harness iyileştirmelerini eşit bütçe ve held-out testle yeniden değerlendiriyor | Prompt/harness ayarı yalnız dev'de; testte ayar yok. Bütçe eşit güçlü baseline'lar şart |
| G3 | Chen ve ark., **AgentCompass** [A] [2607.13705](https://arxiv.org/abs/2607.13705) | 15 Tem 2026 | Benchmark, harness ve ortamı ayıran değerlendirme altyapısı | Veri / yöntem / skorlama arayüzlerini ayırmamızı destekliyor (bizdeki builder / runner / scorer ayrımı) |
| G4 | Zhang ve ark., **When Agents Fail: Model or Harness — Who's to Blame?** [A] [2610.32483](https://arxiv.org/abs/2610.32483) | 23 Eki 2026, under review | Hataları model ve harness etkisine ayırıyor; geçerli tamamlanma ve token yükünü raporluyor | Doğruluk ile tamamlanma ayrı raporlanmalı; bir harness'in tek modeldeki sonucu genellenmemeli |
| G5 | Bertalanič ve Fortuna, **The Cost of Consensus: Isolated Self-Correction Prevails Over Unguided Homogeneous Multi-Agent Debate** [A] [2605.00914](https://arxiv.org/abs/2605.00914) | 29 Nis 2026 | Homojen tartışmayı izole öz-düzeltme kontrolüyle karşılaştırıyor; izole düzeltme kazanıyor | Bizdeki `isolated_refine` ve `debate` kolları tam bu karşılaştırma; dermatolojide tekrar ediyor mu sorusu |
| G6 | Ferreira, Liu ve Zheng, **Beyond Symmetric Agents: Cognitive Diversity and Multi-Agent Debate in Small Language Models** [A] [2609.35875](https://arxiv.org/abs/2609.35875) | 26 Eyl 2026, ICLR 2027'ye gönderilmiş | Persona, sıcaklık ve model çeşitliliği; üretim bütçesi eşit oylama kontrolleri | `role_vote`–`vote3` karşılaştırmamızın genel karşılığı; rol etkisini model gücünden ayırmak |
| G7 | Zhao ve ark., **When Debate Helps: Proposal Supply and Verification-Aware Readout** [A] [2610.04686](https://arxiv.org/abs/2610.04686) | 3 Eki 2026 | Doğru adayın havuzda olmasıyla doğru adayın seçilmesini ayırıyor | İyileşen/bozulan karar geçişlerini biz icat etmedik; atıf yapılarak kullanılmalı |
| G8 | **Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents** [A] [2607.04528](https://arxiv.org/abs/2607.04528) | 5 Tem 2026 | Görev, ortam ve model sabitken harness'in ajanın "inançlarını" değiştirdiğini gösteriyor (kod ajanları) | Aynı modelin farklı harness'te farklı karar vermesi beklenen bir olgu; bizim geçiş analizine kavramsal destek |

## 2. Medikal ajan çalışmaları

| # | Çalışma | Tarih / durum | Ne yapıyor | Bizim için anlamı |
|---|---|---|---|---|
| **M0** ★ | Wang, Zhao ve Ding, **MedicalHarness: A Controlled Evaluation of LLMs and Agent Harnesses on Medical Tasks** [A] [2610.05778](https://arxiv.org/abs/2610.05778) | 5 Eki 2026 | 5 açık model × 5 harness, 107 medikal görev. Bağlam yönetimi, planlama ve araç erişimini tek tek kapatan MH-Lab ile varyans ayrıştırması. Harness ve etkileşim yaklaşık %25 | **En yakın çalışma.** Farkımız: (i) tek ajanlı araç harness'leri değil, çok ajanlı akıl yürütme harness'leri (rol, tartışma, moderatör) ve çağrı sayısı eşit kontroller; (ii) alana özgü tanı (dermatoloji); (iii) hasta beyanı metni ve 2. aşamada aynı vakalara görüntü. Varyans ayrıştırma analizini biz de uygulamalıyız |
| **M0b** ★ | Tang ve Wang, **MedHarnessBench: A Unified Benchmark for Healthcare Agent Evaluation** [A] [2610.15714](https://arxiv.org/abs/2610.15714) | 11 Eki 2026 | FHIR/EHR ve araç görevlerinde (342 görev) sonucun yanında 7 süreç boyutu | Süreç ve iz kaydının önemini destekliyor; bizim görevimiz statik tanı, EHR işlemi değil |
| M1 | Qian ve ark., **MedMASLab**: benchmarking multimodal medical multi-agent systems [A] [2603.09909](https://arxiv.org/abs/2603.09909) | 10 Mar 2026 | 11 çok ajanlı mimariyi ortak protokolle ve çağrı bazında token/süre muhasebesiyle karşılaştırıyor | Yöntemsel olarak en yakın ikinci referans; ajan sayısını artırmak her görevde yardımcı olmuyor |
| M2 | Martinez, **Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA** [A] [2603.24481](https://arxiv.org/abs/2603.24481) | 25 Mar 2026 | Uzman ajanlar ve tutarlılık doğrulamasıyla kalibrasyon (MedQA/MedMCQA alt kümeleri) | Ajanlar arası anlaşma güven olasılığı değildir; kalibrasyon ayrıca ölçülmeli |
| M3 | Liu ve ark. (Kather grubu), **Benchmarking LLM-based agent systems for clinical decision tasks**, *npj Digit Med* 9 [P] [DOI](https://doi.org/10.1038/s41746-026-02443-6) | 18 Şub 2026, yayın | OpenManus ve Manus ile AgentClinic, MedAgentsBench ve HLE. Mütevazı kazanç, >10× token, >2× gecikme | Maliyeti ana tabloya koymanın ve güçlü tek model baseline'ının gerekçesi |
| M4 | Zhao ve ark., **An agentic system for rare disease diagnosis with traceable reasoning (DeepRare)**, *Nature* 651(8106):775–784 [P] [DOI](https://doi.org/10.1038/s41586-025-10097-9) | 18 Şub 2026, yayın (2025 preprint'in yayın sürümü) | Serbest metin fenotip, HPO ve genetik veriyle izlenebilir tanı ajanı | Metinden tanı ve izlenebilirlik örneği; dermatoloji değil |
| M5 | Schmidgall ve ark., **AgentClinic: a multimodal benchmark for tool-using clinical AI agents**, *npj Digit Med* 9 [P] [DOI](https://doi.org/10.1038/s41746-026-02674-7) | 27 Nis 2026, yayın | Eksik bilgi altında hasta etkileşimi ve araç kullanımı; sıralı formatta doğruluk çok düşüyor | Statik vignette ile etkileşimli simülasyon farklı görevler; biz statik görevi açıkça adlandırıyoruz |
| M6 | Waisberg ve Guarnieri, **Agentic AI in Medicine: … the Case for Clinical Testing Harnesses**, *Ann Biomed Eng* 54 [P] [DOI](https://doi.org/10.1007/s10439-026-04381-6) | Eyl/Kas 2026, mektup | Klinik risk birimi tek çıktı değil eylem dizisidir. Senaryo kütüphanesi, tam iz gözlemi ve kademeli kanıt eşiği öneriyor | Hocanın "harness for medical cases" fikrine kavramsal olarak en yakın tıp yazısı; iz kaydımızı gerekçelendirir |
| M7 | Wang ve Yang, **The Consistency Illusion: How Multi-Agent Debate Hides Reasoning Misalignment** [A] [2606.08457](https://arxiv.org/abs/2606.08457) | 7 Haz 2026 | Uzlaşan ajanların gerekçelerinin uyumsuz olabileceğini gösteriyor; olgu temelli tartışma protokolü öneriyor | "Uzlaştılar, demek ki doğru" varsayımına karşı; gerekçe analizi için referans |
| M8 | Hou ve ark., **SMART-Med**: selective multi-agent medical reasoning via uncertainty-aware routing [A] [2610.33544](https://arxiv.org/abs/2610.33544) | 24 Eki 2026, NeurIPS notu | Belirsizliğe göre tek cevap ya da uzman paneli arasında öğrenilmiş yönlendirme | Belirsizlikte uzman çağırmak yeni değil; ileride "adaptive" kol eklenirse öncül |
| M9 | Ge ve ark., **ClinicalAgents**: orchestration with dual-memory [A] [2603.26182](https://arxiv.org/abs/2603.26182) | 27 Mar 2026, KDD | Hipotez arama ve çalışma/deneyim belleği | Test sırasında bellek güncellemesi bizim deneyde kapalı (sızıntı riski) |
| M10 | Xia ve ark., **A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support** [A] [2609.05069](https://arxiv.org/abs/2609.05069) | 4 Eyl 2026 | Roller, tartışma ve birleştirme ile zor olgu tanısı | DermArbiter'in rol + moderatör tasarımı tek başına özgün değil |
| M11 | Riccio ve ark., **Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision** [A] [2611.05393](https://arxiv.org/abs/2611.05393) | arXiv'de 14 Eyl 2026 | Tartışmada çıkan boşluğa göre bilgi getirme veya uzman daveti | "Skeptic gerekince RAG çağırsın" fikriyle güçlü örtüşme |
| M12 | Wasi ve ark., **Is There a Dr. House in the Machine?** (survey), Findings IJCNLP-AACL 2026, s. 1632–1667 [O] [ACL](https://aclanthology.org/2026.findings-aacl.82/) | Kas 2026 | Medikal ajanlarda akıl yürütme, mimari ve araç ekseninde sınıflandırma | Giriş bölümü taksonomisi için |
| M13 | Worth ve ark., **State of clinical AI in 2026**, *BMJ Digit Health AI* [P] [DOI](https://doi.org/10.1136/bmjdh-2026-000113) | Eyl 2026, derleme | Benchmark doygunluğu, kalibrasyon zaafları, çok ajanlı ve multimodal eğilimler | Giriş ve tartışma için güncel bağlam |

## 3. Dermatoloji ajanları ve yakın değerlendirme çalışmaları

Not: Bu bölümdeki her kayıt ajan sistemi değildir. Ajanlar (D1–D4), benchmark ve veri çalışmaları (D5–D9) ve görüş/derleme yazıları (D10–D14) ayrı tutuldu.

**Ajan sistemleri**

| # | Çalışma | Tarih / durum | Ne yapıyor | Bizim için anlamı |
|---|---|---|---|---|
| D1 | Liu ve ark., **DermAgent**: self-reflective agentic system for dermatological image analysis [A] [2605.14403](https://arxiv.org/abs/2605.14403) | 14 May 2026, MICCAI 2026 | Plan, uygula ve yansıt döngüsünde 7 görsel/dil aracı, retrieval ve critic | Görüntü merkezli; metin-only uyarlama yapılırsa "DermAgent-inspired" diye adlandırılmalı |
| D2 | Chen ve ark., **SkinGPT-X**: self-evolving collaborative multi-agent system [A] [2603.26122](https://arxiv.org/abs/2603.26122) | 27 Mar 2026 | Görsel ve tanı ajanları, bilgi getirme ve gelişen bellek | "Dermatoloji + çok ajan + RAG" birleşimi tek başına özgünlük değil |
| D3 | Hao ve ark., **SkinMind**: agentic RAG with human-inspired graph probing [A] [2611.08935](https://arxiv.org/abs/2611.08935) | 6 Kas 2026 | Bilgi grafiğinde çok düzeyli kanıt arayan 8 araçlı orkestratör, 6 VQA benchmark'ı | Adaptif kanıt arama fikrine çok yakın; RAG kolumuz sabit ve açık kaynaklı kalmalı |
| D4 | **DERM-3R**: resource-efficient multimodal agents for dermatologic diagnosis and treatment [A] [2604.09596](https://arxiv.org/abs/2604.09596) | 6 Mar 2026 | Tanıma, temsil ve akıl yürütme ajanlarıyla Geleneksel Çin Tıbbı dermatoloji karar desteği | Farklı tıbbi çerçeve; yalnız ilgili çalışma olarak anılmalı |

**Benchmark ve veri çalışmaları**

| # | Çalışma | Tarih / durum | Ne yapıyor | Bizim için anlamı |
|---|---|---|---|---|
| D5 | **Yilmaz ve ark., DermaBench**: clinician-annotated benchmark for dermatology VQA and reasoning [A] [2601.14084](https://arxiv.org/abs/2601.14084) | 20 Oca 2026 | DDI üzerinde 656 görüntü, 22 soruluk klinisyen anotasyon şeması | **Hocanın çalışması.** 2. aşamada (görüntü) doğal köprü; DermArena şeması buna hizalanmış (`docs/query_suite_alignment.md`) |
| D6 | Liu ve ark., **DermCase** — *Mind the Rarities: Can Rare Skin Diseases Be Reliably Diagnosed via Diagnostic Reasoning?* [A] [2603.18418](https://arxiv.org/abs/2603.18418) | 19 Mar 2026 | Olgu sunumlarından 6.354 nadir hastalık vakası; tanı, ayırıcı tanı ve akıl yürütme | Uzun klinik metin içeriyor: Track C (açık uçlu olgu) için en iyi aday, erişim ve lisans kontrolü gerekli |
| D7 | Ru ve ark., **DermoGPT / DermoBench** [A] [2601.01868](https://arxiv.org/abs/2601.01868) | 5 Oca 2026 | Morfoloji temelli model, veri ve 11 görevli benchmark | "Dermatolojide çok boyutlu değerlendirme yok" denemez; DermaBench ile karıştırılmamalı |
| D8 | Zhao ve ark., **DermEVAL**: dermatologist-reviewed benchmark for MLLMs, WACV 2026 [W] [CVF](https://openaccess.thecvf.com/content/WACV2026/html/Zhao_DermEVAL_A_Dermatologist-Reviewed_Benchmark_for_Multimodal_Large_Language_Models_WACV_2026_paper.html) | Mar 2026, yayın | 16 hastalık, 11.347 görüntüde VQA ve rapor üretimi | Görsel görev; görüntüyü çıkarmak geçerli bir metin benchmark'ı yapmaz |
| D9 | Nouyed ve ark., **Text-dominant decision-making by large multimodal models in dermatology clinical challenges**, *JAAD* 95(4) [P] [DOI](https://doi.org/10.1016/j.jaad.2026.05.129) | Haz/Eki 2026, mektup | 44 JDCR vakasında metin, görüntü ve birleşik koşul; modellerin metne dayandığını öne sürüyor | **Metin-only dermatoloji için doğrudan öncül.** Bizim "önce text, sonra image" tasarımımızı destekliyor. Aynı vakadan gelen soruların bağımsız sayılmaması dersi |
| D10 | Yoo, Kim ve Rosen, **Beyond Diagnosis**: recommendation stability under counterfactual skin-tone perturbations, Findings IJCNLP-AACL 2026, s. 4549–4564 [O] [ACL](https://aclanthology.org/2026.findings-aacl.230/) | Kas 2026 | 60 vakadan 240 hekim doğrulamalı görüntüde (vaka başına 4 cilt tonu varyantı) öneri kararlılığı; öneri değişimi eşleştirilmiş karşılaştırmaların %51,7'sine kadar | 2. aşama adalet deneyleri için örnek; metin aşamasına taşınmaz |

**Görüş, derleme ve çerçeve yazıları (PubMed)**

| # | Çalışma | Tür / tarih | Bizim için anlamı |
|---|---|---|---|
| D11 | Yan, Mar, Kittler ve Ge, **Toward agentic AI co-practitioners in dermatology**, *Cell Rep Med* 7 [P] [DOI](https://doi.org/10.1016/j.xcrm.2026.103171) | Derleme, 17 Kas 2026 | PanDerm ekibinden. "Uçtan uca doğrulanmış bir dermatoloji ajanı henüz yok" ve "retrospektif benchmark yetmez" diyor. Sınırlılıklar bölümümüzde mutlaka atıf |
| D12 | Chu ve ark., **Agentic AI in Dermatology: A Call to Action**, *JMIR Dermatol* 9 [P] [DOI](https://doi.org/10.2196/94909) | Makale, Haz 2026 | Klinik risk ve geri döndürülebilirliğe göre risk sınıflandırması |
| D13 | Kim, Rios-Duarte ve Vidal, **AI agents assemble: The new age of AI agentic workflows and their clinical applications**, *JAAD* 95(3) [P] [DOI](https://doi.org/10.1016/j.jaad.2026.04.2007) | Makale, May/Eyl 2026 | Dermatolojide ajan iş akışlarına giriş |
| D14 | Kantor, **Agentic AI in medicine and biomedical research: A primer**, *JAAD Int* 27 [P] [DOI](https://doi.org/10.1016/j.jdin.2026.06.001) ve **The confidence game**, *JAAD Int* 29 [P] [DOI](https://doi.org/10.1016/j.jdin.2026.10.001); Goldust, **Agentic AI in Dermatology**, *Int J Dermatol* 65 [P] [DOI](https://doi.org/10.1111/ijd.70446) | Editoryal ve mektuplar, 2026 | Otomasyon yanlılığı, hata yayılımı ve insan denetimi: tartışma bölümü bağlamı |

İlgili ama ajan değil: Li ve ark., **MAGEN / O-MAKE** (*IEEE TMI* 45, Kas 2026) [P] [DOI](https://doi.org/10.1109/TMI.2026.3723004). Çok ajanlı *veri üretimi* ile Derm1M'i zenginleştiriyor; bizdeki Derm1M ve MAKE araçlarıyla bağlantılı.

## 4. Konumlandırma: neyi iddia edebiliriz?

| Olası iddia | En yakın öncül | Karar |
|---|---|---|
| "Dermatolojiye çok ajanlı sistem kurduk" | D1, D2, D3 | ❌ Özgün değil |
| "Harness'in medikal sonuçları değiştirdiğini gösterdik" | M0 MedicalHarness | ❌ Genel olarak gösterildi; alan içinde *tekrar* ediyoruz |
| "Tartışma tek ajandan iyidir" | G5, G7, M3 | ❌ Öncül bulgular karşı yönde; sonucu deney belirleyecek |
| "Metin-only dermatoloji değerlendirdik" | D9 | ⚠️ Tek başına zayıf |
| **"Aynı backbone ve aynı vaka metni altında, çağrı sayısı eşit kontrollerle 10 harness'i (tek çağrı → rol paneli + tartışma + moderatör) iki backbone ailesinde, hasta beyanı (dermatolog etiketli) ve sınav metni üzerinde karşılaştırdık; kazançları, maliyeti ve karar geçişlerini raporladık"** | M0 + G5–G7 + D9 birleşimi | ✅ **Savunulabilir ana katkı** |
| "Aynı vakalarda 2. aşamada görüntü eklendiğinde harness etkisinin değişimi" | Bulunamadı | ✅ Gelecek çalışma (2. aşama) için güçlü hook |

**Makaleye doğrudan geçecek yöntem dersleri**

1. Çağrı ve token sayısı eşit kontroller: `vote6` ve `isolated_refine` kolları `debate` ile karşılaştırılmalı (G5, G6, M1).
2. Model × harness etkileşimi: en az iki backbone ailesi ve MedicalHarness tarzı varyans ayrıştırması (M0).
3. Maliyet ana tabloda: çağrı, token ve süre (M3, M1).
4. Tamamlanma ile doğruluk ayrı; tüm vakalar paydada (G4).
5. Anlaşma, güven olasılığı değildir (M2, M7).
6. İyileşen/bozulan geçişler G7'ye atıfla raporlanır.

## 5. Dışarıda bırakılanlar ve doğrulanacaklar

* *Towards Trustworthy Dermatology MLLMs* ([2511.09195](https://arxiv.org/abs/2511.09195)): ilk gönderim Kasım 2025. ECCV 2026 sürümü eşleşirse eklenebilir.
* DeepRare'in 2025 preprint'i ([2506.20430](https://arxiv.org/abs/2506.20430)): yalnız 2026 *Nature* sürümü sayıldı (M4).
* **MMedAgentBench** (Sun ve ark., EMNLP 2026, s. 10780–10793 [O] [ACL](https://aclanthology.org/2026.emnlp-main.469/)): çok aşamalı klinik karar verme. Ajan, yeni tetkik sonuçları geldikçe kararını güncelliyor. Üçüncü basamak hastanelerden 150'den fazla anonim vaka, 500'den fazla klinik aşama, uzman hekim anotasyonu var. Tanı doğruluğu ile kaynak/maliyet verimliliğini birlikte ölçüyor. Özette dermatoloji, multimodal içerik ve karşılaştırılan ajan sayısı geçmiyor. Bizimle ilişkisi: tek aşamalı metin girdisinde harness karşılaştırıyoruz; onların ölçtüğü "tetkik planlama + maliyet" ekseni bizde yok. Tartışma bölümünde gelecek iş olarak anılabilir, doğrudan rakip değil.
* **SkinAgent AI** ([2609.29341](https://arxiv.org/abs/2609.29341)): tanı amaçlı olmayan cilt bakımı ajanı; kapsam dışı, gerekirse bir cümleyle anılır.
* 22 Kasım kontrolü: M4 PubMed'den (PMID 41708847), M12 ve D10 ACL Anthology'den doğrulandı. D8 (DermEVAL) için CVF sayfası otomatik erişimi reddetti (HTTP 403); 16 hastalık ve 11.347 görüntü sayıları tarayıcıdan elle kontrol edilmeli.

## 6. Güncelleme planı

* 20 Aralık: arXiv `cs.AI`/`cs.CL` + "medical agent", "harness", "multi-agent debate", "dermatology LLM" ve PubMed "agentic AND dermatology" için tekrar tarama.
* MedicalHarness'ın kod ve verisi incelenecek (`github.com/REAL-Lab-NU/MedicalHarness`). Varyans ayrıştırma yöntemini birebir uygulamak karşılaştırılabilirliği artırır.
