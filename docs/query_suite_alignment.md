# Soru Demeti Hizalaması — DermaVQA-DAS & DermaBench'e Dayandırma

> **Amaç:** Klinik soru demetimizi (`query_suite.py`, 11 tip) bizim elle
> yazdığımız doğrulanmamış şablonlar olmaktan çıkarıp, **klinisyen-valide
> yayınlanmış şemalara dayandırmak** — böylece içerik geçerliliği (content
> validity) + karşılaştırılabilirlik kazanmak.

## Sorun

Mevcut 11 soru tipi (`QUERY_TEMPLATES`) ekip tarafından elle yazıldı. Klinik
akıl yürütme akışına dayanıyorlar ama harici bir standartla **valide
edilmediler**. Nature Medicine reviewer'ı "bu sorular kim tarafından,
hangi temele göre yazıldı?" diye sorar.

## İki dış kaynak (klinisyen-valide, yayınlanmış)

| | DermaBench (arXiv 2601.14084, Oca 2026) | DermaVQA-DAS (arXiv 2512.24340, Ara 2025) |
|---|---|---|
| Şema | 22 soru (Q0-Q21), 3 blok | DAS: 72 üst-düzey / 137 ince soru, 9 kategori |
| Veri | **DDI** (656 görüntü, 570 hasta, Fitz I-VI) | patient-generated; closed-QA + segmentasyon |
| Anotasyon | 6 dermatolog, konsensüs (AnnotatorMed) | 3 anotatör, çoğunluk oyu; EN+ÇN |
| Skorlama | (makalede belirtilmemiş) | çoktan-seçmeli doğruluk (o3 0.798, GPT-4.1 0.796) |

### Neyi ölçüyorlar? — **morfoloji betimleme**
Her ikisi de ağırlıklı olarak **"lezyon nasıl görünüyor?"**:
- DermaBench Q11-Q20: sayı, lokasyon, dağılım, şekil, kenar, yüzey, boyut,
  renk, elementer lezyon tipleri
- DermaVQA-DAS: anatomik lokasyon, boyut, renk, morfoloji, eşlik eden semptom

### Neyi ölçmüyorlar? — **klinik karar & agentic davranış**
Tanı sonrası karar zinciri (triyaj, yönetim, kodlama, risk), kalibrasyon,
hasta iletişimi, ve **kapsam/güvenlik sınırları** — bunlar onlarda yok.

## Örtüşme haritası (bizim 11 tip → dış şema)

| Bizim soru tipi | Dış şemada karşılığı | Aksiyon |
|---|---|---|
| `DERMOSCOPIC` (morfoloji) | **DermaVQA-DAS DAS** + **DermaBench Q11-Q20** | ✅ **BENİMSE** — kendi tek serbest-metin sorumuz yerine onların yapılandırılmış morfoloji şemasını kullan |
| `DIAGNOSIS`, `DIFFERENTIAL` | kısmen (tanı bağlamı) | hizala, kodları/etiketleri eşle |
| `TRIAGE`, `MANAGEMENT`, `RISK_ASSESSMENT` | ❌ yok | **bizim katkımız** — koru |
| `CODING` (ICD/SNOMED) | ❌ yok | bizim katkımız — koru |
| `HISTORY_ANALYSIS` | kısmen (semptom) | hizala |
| `TREATMENT`, `PATIENT_COMMUNICATION` | ❌ yok | bizim katkımız — koru |
| `COMPARATIVE` | ❌ yok | bizim katkımız — koru |
| **Boundary/Scope (Dim 9)** | ❌ yok | bizim katkımız — koru |

## Önerilen plan

1. **Morfoloji katmanını dışarıdan al.** `DERMOSCOPIC` sorusunu, DermaVQA-DAS'ın
   **DAS şemasıyla** (anatomik lokasyon / boyut / renk / morfoloji / semptom)
   değiştir. Bu, Dim 1-2'ye (visual + narrative) **valide, çoktan-seçmeli,
   karşılaştırılabilir** bir omurga verir. DDI tabanlı çalıştığımız için
   DermaBench'in DDI anotasyonlarıyla da **doğrudan hizalanabiliriz**.
2. **Klinik-karar katmanını koru ve sahiplen.** Triyaj, yönetim, kodlama,
   risk, tedavi, iletişim — literatürde bu yapıda yok; bizim özgün eksenimiz.
3. **Agentic + sınır katmanını öne çıkar.** Dim 9 (scope/safety) ve çok-ajan
   tartışma değerlendirmesi → asıl farkımız.
4. **Atıf + pre-registration.** Soru şemasını yayınlardan türettiğimizi
   açıkça belirt (Related Work + Methods), seti eval öncesi dondur.

## Net sonuç

Soru demetimiz **icat değil, hizalama** olmalı: morfoloji kısmını
klinisyen-valide şemalardan (DermaVQA-DAS/DermaBench) devral, klinik-karar +
agentic + sınır kısmını **bizim özgün katkımız** olarak konumlandır. Böylece
hem geçerlilik kazanırız hem özgünlüğümüz netleşir.

## Kaynaklar
- DermaBench: https://arxiv.org/abs/2601.14084
- DermaVQA-DAS: https://arxiv.org/abs/2512.24340
- DermaVQA (MICCAI 2024): https://link.springer.com/chapter/10.1007/978-3-031-72086-4_20
