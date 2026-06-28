# İsimlendirme & Konumlandırma — DermaBench/DermaVQA/DermAgent'a Karşı

> **Neden bu doküman:** Benchmark'ımızın çalışma adı "DermAbench", Ocak 2026'da
> yayınlanan **DermaBench** ile çakışıyor (hem isim hem konsept: DDI tabanlı,
> klinisyen-anotasyonlu, derm VQA). İsim değişmeli ve özgünlüğümüz net
> konumlanmalı. Bu doküman makale §Related Work + isim kararı için temeldir.

---

## 1. İsim değişikliği (zorunlu)

"DermAbench" ↔ "DermaBench" karışıklığı + öncelik tartışması doğurur.
Ayrıca "DermAgent" (karşılaştırdığımız sistem) ve "DermaVQA" da dolu.

### Aday isimler (çakışmasız, özümüzü yansıtan)

| Aday | Gerekçe | Risk |
|---|---|---|
| **DermArena** ⭐ | Çok-ajan **tartışma arenası**nı çağrıştırır; sistemimiz DermArbiter ile uyumlu | düşük |
| **DermTrust** | Güvenilirlik eksenleri (kalibrasyon+fairness+safety+scope) | jenerik olabilir |
| **A-DERM** | "Agentic Dermatology Evaluation of Reasoning & Management" kısaltması | telaffuz |
| **DELPHI-Derm** | Delphi = uzman paneli/konsensüs; agentic debate metaforu | niş |

**Öneri: DermArena** — multi-agent agentic değerlendirmeyi en iyi yansıtan,
çakışmasız, akılda kalıcı ad. (Nihai karar Emre+Furkan+Abdurrahim'in.)

---

## 2. Manzara: üç ilgili çalışma

| | Ne yapıyor | Eksik bıraktığı |
|---|---|---|
| **DermAgent** | Agentic derm sistemi + kendi benchmark'ı | trustworthiness eksenleri, sınır/güvenlik, confound-kontrollü ablation |
| **DermaBench** (Oca 2026) | DDI'da morfoloji VQA, **tek-model**, klinisyen-anotasyonlu 22 soru | klinik karar zinciri, agentic, kalibrasyon, sınır |
| **DermaVQA-DAS** (Ara 2025) | Yapılandırılmış morfoloji closed-QA + segmentasyon | klinik karar, agentic, güvenlik/sınır |

**Ortak boşluk:** Hepsi *statik soru-cevap* / *tek-model* / *morfoloji-betimleme*
odaklı. Hiçbiri **agentic bir sistemin güvenilirliğini ve sınır davranışını**
ölçmüyor.

---

## 3. Bizim 5 farklılaşma direğimiz

1. **Agentic değerlendirme.** Tek modeli değil, **çok-ajan tartışmalı
   sistemi** (4 ajan × 5 faz) sınarız. Kritik: aynı temel modeli
   **scaffold'lu vs scaffold'suz** koşarak agentic katkıyı confound-kontrollü
   izole ederiz (DermArbiter-Gemini vs Gemini-tek-başına).
2. **Klinik karar ekseni.** Morfoloji betimlemenin ötesinde: triyaj, yönetim,
   kodlama (ICD/SNOMED), risk → tam klinik karar akışı.
3. **Güvenilirlik eksenleri.** Kalibrasyon (ECE), **fairness** (Fitzpatrick
   açığı), **safety** (malign triyaj duyarlılığı), **grounding** (kanıt
   atıfı). Bunlar accuracy'nin görmediği yer.
4. **Kapsam & Sınır (Boyut 9).** Klinik dışı / adversarial / over-refusal
   davranışı — literatürde nadir, agentic güvenlik için kritik.
5. **9-boyutlu holistik karne.** Tek sayı değil; sistemin her yeteneği ayrı.

---

## 4. Tek cümlelik konumlandırma (makale için)

> *"DermAgent, DermaBench ve DermaVQA dermatoloji modellerinin **ne kadar
> doğru bildiğini** ölçer; biz agentic bir sistemin **ne kadar güvenilir,
> adil, güvenli ve kapsamında** karar verdiğini ölçen ilk holistik,
> sınır-farkında benchmark'ı sunuyoruz."*

---

## 5. İlişki: rakip değil, üstüne inşa

- **Morfoloji katmanı:** DermaVQA-DAS şemasını **devralırız** (bkz.
  [query_suite_alignment.md](query_suite_alignment.md)) → karşılaştırılabilirlik.
- **Veri:** DDI'ı paylaşıyoruz (DermaBench de DDI tabanlı) → sonuçlarımız
  doğrudan kıyaslanabilir.
- **Katkı:** Onların morfoloji/VQA tabanına **agentic + güvenilirlik + sınır**
  katmanlarını ekleriz.

Bu, "yeniden icat" değil, mevcut valide tabana **agentic değerlendirme
omurgası** eklemek — hem güvenli hem özgün.

## Kaynaklar
- DermAgent, DermaBench (https://arxiv.org/abs/2601.14084),
  DermaVQA-DAS (https://arxiv.org/abs/2512.24340)
