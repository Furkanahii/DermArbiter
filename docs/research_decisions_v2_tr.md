# DermArbiter: araştırma kararları ve deney protokolü v2

Tarih: 21 Kasım 2026. Bu belge önceki `text_only_protocol_2026_tr.md` içindeki taslak yaklaşım listesini günceller. İlk belgedeki veri sızıntısı, kaynak ayrımı ve test izolasyonu ilkeleri geçerlidir. Aşağıdaki kararlar araştırma önerisidir; henüz toplanmış model başarım sonuçları değildir.

## 1. Makalenin amacı ve sınırı

Önerilen başlık: **DermArbiter: A Controlled Benchmark of Agentic Harnesses for Text-Based Dermatological Reasoning**.

Araştırma sorusu: Sabit temel model ve aynı metin soruları altında, tek cevap, öz değerlendirme, bağımsız örnekleme ve ajanlar arası tartışma doğruluk, tamamlanma ve maliyet bakımından nasıl ayrışıyor? Tartışma başlangıç kararlarını hangi sıklıkla düzeltiyor veya bozuyor?

“Improved” sözcüğü ancak birincil önceden belirlenmiş karşılaştırmada sonuç desteklerse başlığa girecek. “Bir yaklaşımlar kümesini karşılaştırdık” anlatısı hocanın yönlendirmesiyle uyumludur. Daha çok ajan veya daha uzun çıktı tek başına katkı sayılmayacak. Çalışma klinik kullanım onayı, doktorların yerine geçme veya görsel tanı başarısı iddiası içermeyecek.

## 2. Veri stratejisi: iki ayrı görev

**Track A — dermatoloji bilgi/MCQ:** Hazır cevap anahtarlı, görüntü gerektirmeyen, tek cevaplı sorular. Yazılım pilotunun en hızlı yolu. Ölçülen şey seçenekler arasında doğru yanıt seçimidir. Tanı sorusu olmayan tedavi/bilgi soruları “diagnosis accuracy” başlığı altında toplanmaz. Soru türleri ayrı raporlanır.

**Track B — klinik vaka metni:** Kaynaktaki başvuru öyküsünün cevaptan ve tartışma bölümünden ayrıldığı vakalar. JDCR ve DermCase adaydır; veri erişimi ve kullanım koşulları tamamlanmadan kesin test kohortu değiller. Kaynaktaki görüntü çıkarıldığında metin yeterli mi, soru görüntüye doğrudan mı atıf yapıyor, tanı ismi gövdeye sızmış mı kontrol edilir. Açık uçlu tanı için dondurulmuş eşanlamlı eşleştirme veya kör uzman değerlendirmesi gerekir; mevcut MCQ scorer bunun yerine kullanılamaz.

Bu iki track'in doğrulukları tek sayıda birleştirilmez. Track B zamanında hazırlanamazsa makale kapsamı açıkça “dermatology MCQ reasoning” olarak daraltılır; klinik vaka benchmark'ı yapılmış gibi yazılmaz. Bu kapsam değişikliğinin veri yeterliliği açısından 30 Kasım'a kadar kararlaştırılması önerilir.

### Somut MedMCQA veri denetimi

Resmi HF kaynağının `91c6572c454088bf71b679ad90aa8dffcd0d5868` sürümünden validation ve test Parquet dosyaları indirildi; sayılar yerelde hesaplandı:

| Dosya | Toplam soru | `subject_name=Skin` | `single` / `multi` | Cevap durumu |
|---|---:|---:|---:|---|
| validation | 4.183 | 17 | 11 / 6 | Dermatoloji kayıtlarında 0–3 aralığında yayımlanmış anahtar |
| test | 6.150 | 60 | 39 / 21 | Bütün test kayıtlarında `cop=-1`; kullanılabilir gold yok |

Validation SHA-256: `b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021`.
Test SHA-256: `1d308841f0c82df363d3d638b33b69b8abd266b1e615ae5d2607dbb24c70beb1`.

11 tek cevaplı kayıt yalnız `dev/pilot_candidate` olarak dönüştürüldü. Bir klinisyen tarafından yeni doğrulama yapılmadı. Bazı sorularda yazım bozukluğu, soruyla seçeneklerin uyumu veya eski bilgi ihtimali ayrıca incelenmeli. `single` etiketi otomatik klinik kalite güvencesi değildir. Bu örnekler yöntemin input/output, hata ve token kaydını kontrol etmek içindir; örneklem çok küçük olduğu için makale ana sonucu olarak kullanılmayacak.

Kaynağın dataset card'ı Apache-2.0 beyan ediyor. Kaynak sürümü, dosya hash'i ve orijinal split saklandı. Soru kimliği kaynak grubu olarak kullanıldı; gerçek hasta veya orijinal sınav grubu bilgisi yok. Train bölümünden ileride daha büyük altküme seçilirse, “public train-derived evaluation” olarak açıkça adlandırılmalı; model eğitimiyle kontaminasyon olmadığı iddia edilmemeli. [Resmi veri kartı](https://huggingface.co/datasets/openlifescienceai/medmcqa/blob/main/README.md).

### Klinik vaka kaynağı için kabul formu

Her vaka için kaynak URL/DOI, lisans/erişim koşulu, orijinal vaka kimliği, soru türü, yayımlanmış gold'un yeri, metnin nasıl çıkarıldığı, görüntüye bağımlılık ve kontrol eden kişi kaydedilecek. Başlık, tartışma, patoloji sonucu ve yanıt açıklaması yalnız kaynakta soru anında gerçekten verilen bilgilerse input olabilir. Sonradan tanıyı açıklayan bölümler modele verilmez. Gerekli görülen çıkarmalar bir edit log ile tutulur; etiket bilerek soruyu kolaylaştıran yeniden yazım yapılmaz.

Hoca çalışmayı bize bırakmış olsa da uzman onayı gerçekleşmeden “dermatologist-validated” ifadesi kullanılamaz. Hazır yayımlanmış answer key, yeni klinisyen anotasyonu yerine geçirilmez. Vaka uygunluğu için uzman incelemesi sağlanamazsa sınav benchmark'ı olarak kapsam sınırlandırılır.

## 3. Yedi deney koşulu

| Kod / uygulama | Akış | En çok çağrı/vaka | Kontrol ettiği soru |
|---|---|---:|---|
| B0 `direct` | Bir cevap | 1 | Temel performans |
| B1 `structured` | Bulgular/alternatifler talimatıyla bir cevap | 1 | Basit prompt düzenlemesinin etkisi |
| B2 `self_refine` | Bir cevap ve iki öz değerlendirme | 3 | Aynı ajana ek işlem vermenin etkisi |
| B3 `vote3` | Üç bağımsız cevap, deterministik oylama | 3 | Ek örnekleme, iletişim olmadan yeterli mi? |
| B4 `vote6` | Altı bağımsız cevap, deterministik oylama | 6 | Tartışmanın çağrı sayısına denk örnekleme kontrolü |
| B5 `isolated_refine` | Üç bağımsız cevap, her birinin yalnız kendi cevabını bir kez yeniden değerlendirmesi, üç son cevapla oylama | 6 | Tartışmayla aynı akış, akran bilgisi olmadan |
| B6 `debate` | Üç bağımsız cevap, ilk paneli gören üç eşzamanlı mantıksal revizyon, son cevaplarla oylama | 6 | Ajanlar arası bilgi paylaşımının katkısı |

“Eşzamanlı mantıksal” ifadesi, bütün revizyonların aynı dondurulmuş ilk paneli görmesi demektir. API çağrıları şu anda sırayla yapılır; ajan 2, ajan 1'in revizyonunu görmez. Oylamada benzersiz en yüksek oy kullanılır; eşitlikte abstention. Bu kural bütün oylama kollarında aynıdır. Oylardaki anlaşma oranı güven olasılığı olarak kullanılmaz.

İlk aşamada rol/persona çeşitliliği yok: aynı backbone ve aynı temel talimat kullanılıyor. Böylece persona etkisiyle iletişim etkisi karıştırılmıyor. Farklı uzman rolleri, RAG ve heterojen modeller daha sonra ayrı ablation olarak eklenebilir. Mevcut uygulama özgün DermAgent/SkinGPT-X reprodüksiyonu değildir.

**Öncelikli karşılaştırmalar:** B6–B5 (iletişimli/iletişimsiz revizyon) ve B6–B4 (iletişim/bağımsız örnekleme). B2–B3 üç çağrılı kontrol. B0 ve B1 referanslar. Her birini her birine karşı test etmek yerine bu sınırlı liste dondurulacak; çoklu test düzeltmesi önceden seçilecek.

## 4. Bütçe eşitliği: çağrı sayısı token eşitliği değildir

Altı çağrı yapan yöntemlerin toplam token sayısı aynı değildir. Tartışma, akran yanıtlarını yeniden girdi yaptığı için daha fazla input token tüketir. Reasoning modellerinde düşünme tokenları da olabilir. Bu nedenle iki analiz ayrılır:

1. **Çağrı sayısı kontrollü analiz:** Aynı backbone, aynı sıcaklık, aynı çağrı başına çıktı üst sınırı; gerçek kullanım ayrıca raporlanır. Mevcut pilot bu düzeydedir.
2. **Toplam tüketim analizi:** Pilot üzerinden birkaç önceden belirlenmiş bütçe noktası oluşturulur; accuracy–gerçek token ve gerekiyorsa fiyatlandırılmış maliyet eğrisi çıkarılır. Yalnız aynı çağrı sayısına bakıp “eşit maliyet” yazılmaz.

Yeni runner provider'ın `usageMetadata` alanını saklar. Toplam token gelmezse sıfır yazmaz, unknown bırakır. HTTP hatalarında sağlayıcı faturalaması bilinmeyebilir. Dolar maliyeti hesaplanacaksa tarihli fiyat tablosu, cache ve düşünme tokenlarının ücretlendirmesi ayrıca doğrulanır; mevcut kod dolar bütçe sınırı uygulamaz.

Tüm yedi kol toplam 26 çağrı/vaka üst sınırı taşır. 11 aday soruyla tek model/tek tekrar planı 286 çağrıdır; bu çağrılar yapılmadı. Örnek olarak 200 soru × 2 backbone × 3 tekrar × 26 = 31.200 çağrı. Bu bir örnek plan, istatistiksel örneklem yeterliliği hesabı veya satın alma onayı değildir. Önce küçük pilotla token/süre dağılımı ölçülür.

## 5. Ölçümler ve istatistik

Birincil ölçüm: planlanan bütün test soruları üzerinde exact option accuracy. API hatası, parse hatası, eksik çıktı ve abstention başarı sayılmaz; ayrı kategorilerle raporlanır. Valid completion ve cevap kapsaması ayrı ölçülür. Yalnız cevap dönen vakaları payda yapmak yasaktır.

İkincil ölçümler: input/output/thinking/total kullanım metadatası, seri çalışmadaki uçtan uca süre, tamamlanma, abstention ve doğru→yanlış / yanlış→doğru karar geçişleri. Başlangıç→son geçişleri betimleyicidir; tek başına nedensel iletişim etkisi göstermez. Aday havuzunda doğru cevap var mı metriği yalnız evaluator tarafında hesaplanabilir; harness gold'a bakarak karar veremez. Bu analizlerin öncülleri genişletilmiş literatür raporunda belirtilmiştir.

Güven analizi ayrı kalacak. Tek modelin söylediği confidence ampirik kalibrasyon kanıtı değildir. Oylama confidence üretmiyor. MCQ için tam seçenek olasılıkları istenmeden çok sınıflı Brier skoru hesaplanmaz. Seçilmiş cevabın doğruluğu üzerinden binary Brier yapılırsa bu tanım ve yalnız uygun kollar kullanıldığı açıkça yazılır. Şu an yeni scorer kalibrasyon skoru üretmiyor.

Vaka raporundan birkaç soru geliyorsa bootstrap birimi soru değil kaynak vaka olmalıdır. Birden çok tekrar ve aynı vakadaki varyantlar bağımsız hasta gibi sayılmaz. Birincil delta için vaka kümeleri üzerinden paired bootstrap; tek bağımsız yanıtlık uygun tabloda McNemar; tekrarlar için vaka içinde tekrarı koruyan değerlendirme planı hazırlanır. Mevcut scorer bu istatistikleri henüz uygulamıyor; pilot öncesi değil ana deney öncesi tamamlanacak.

İki temel model, makul biçimde farklı model ailelerinden seçilecek. Üç tekrar başlangıç önerisi; belirsizlik ve bütçeye göre dev aşamasında karar verilecek. Aynı seed API'de deterministik tekrar garantisi değildir. Yeni kod ilk çağrılara eşleştirilmiş seed veriyor fakat yöntemleri ayrı çalıştırmak aynı ilk yanıt havuzunu garanti etmez. Sabit-proposal nedensel analiz yapılacaksa ilk cevapların ortak kaydedilip farklı revizyon koşullarına yeniden verildiği ayrı bir fork deneyine ihtiyaç var; mevcut pilot böyle bir analiz yaptığını iddia etmiyor.

## 6. Yenilik için kontrollü seçenek

Ana katkı benchmark ve hata analizi olacak. Opsiyonel bir “vaka bulgusuna bağlı revizyon” kolu ancak en yakın GDP/verification çalışmalarından farkı açıkça gösterilirse geliştirilir. Örneğin her değişen cevabın hangi vaka bulgusu veya hangi çelişki nedeniyle değiştiği denetlenebilir; bunun klinik doğruluğu uzman rubriği olmadan otomatik doğrulanmış sayılmaz.

Geliştirme yalnız dev bölümünde yapılır. Testte kötü sonuç görünce eşik veya prompt değiştirilmez. Yeni kol, B5/B6 ve aynı bilgiye sahip tek ajan kontrolüne karşı değerlendirilir. Etkisi belirsiz ya da maliyeti yüksekse makaleden üstünlük iddiası çıkarılır; benchmark çalışması devam eder. Bu yöntemin özgünlüğü veya yararı şu an kanıtlanmış değildir.

## 7. Yıl sonu takvimi ve karar kapıları

| Tarih | Somut çıktı | Devam ölçütü |
|---|---|---|
| 21–24 Kasım | Literatür v2, yakın çalışma matrisi, çalışan MCQ pilot hattı | Yanlış novelty iddiaları elenmiş; testler geçiyor |
| 25–30 Kasım | Kaynak/lisans ve veri kalite denetimi; task etiketleri; kaynak bazlı dev/test taslağı | Ana kohort yeterli ve erişilebilir; değilse kapsam daralır |
| 1–6 Aralık | Küçük dev pilotu; iki backbone seçimi; gerçek token/süre; prompt ve bütçe dondurma | Parse/erişim sorunları çözülmüş; test kullanılmamış |
| 7–13 Aralık | Ana deneyler ve önceden belirlenen tekrarlar | Tam manifest/iz/başarısızlık kayıtları |
| 14–20 Aralık | Paired analizler, hata grupları, gerekirse kör uzman incelemesi | Belirsizlik ve sınırlamalar sonuçlarla birlikte raporlanmış |
| 21–27 Aralık | Makale, tablolar/figürler, reprodüksiyon paketi | İddialar veriye bağlı; literatür güncel |
| 28–31 Aralık | Son kontrol ve uygun hedefe gönderim hazırlığı | Yazarlık/format/veri erişimi tamam |

Yıl sonu hedefi tamamlanmış makale ve gönderime hazır pakettir; kabul tarihi kontrolümüzde değildir. Ana veri kohortu hazır değilken belirlenmiş takvimi korumak için sahte veya otomatik üretilmiş klinik ground truth kullanılmayacak.

## 8. Bu tur tamamlananlar ve kalanlar

Tamamlanan: 28 odak çalışmalı iki rapor; yedi MCQ koşulunun uygulaması; doğrudan Gemini REST adaptörü; model sürümü kontrolü; raw output/prompt/seed/token/süre kaydı; cevap anahtarından ayrılmış input; tüm planlanan vakalar üzerinden skor; MedMCQA'nın iki split'inde somut veri denetimi; 11 aday dev sorusu ve çağrı yapmayan plan.

Kalan: kabul edilmiş ana test kohortu, klinik kapsam kontrolü, canlı API smoke testi, ikinci sağlayıcı adaptörü, fiyatlandırılmış bütçe, paired güven aralıkları, RAG koşulları ve gerekirse açık uçlu scorer. API anahtarı okunmadı, ücretli model çağrısı yapılmadı. Dolayısıyla henüz bir başarı yüzdesi, “en iyi yöntem” veya makale sonucu yok.
