# MedScan-AI
# 🏥 MedScan-AI: Medical Image Analysis Pipeline

MedScan-AI, göğüs röntgeni (CXR) görüntülerinden otomatik hastalık teşhisi (COVID-19, Pnömoni) gerçekleştiren ve doktorlar için dijital raporlama sunan bir yapay zeka sistemidir.

## 🚀 Öne Çıkan Özellikler & İnovasyon (USP)

- **Multiple Batch Processing:** Sisteme aynı anda onlarca röntgen yükleyebilir, saniyeler içinde toplu analiz sonuçları alabilirsiniz.
- **Triage (Önceliklendirme) Sistemi:** COVID veya Pnömoni saptanan vakalar, doktorun iş akışını hızlandırmak için otomatik olarak "YÜKSEK ÖNCELİKLİ" bayrağıyla işaretlenir.
- **Metadata Stripping:** Hasta gizliliği ve etik AI ilkeleri gereği, yüklenen tüm görüntülerdeki EXIF ve kişisel veriler işleme aşamasında otomatik olarak temizlenir.

## 🛡️ Güvenlik ve Hata Yönetimi (Kılavuz 3.4 Uyumluluğu)

Jüri değerlendirme kriterlerine uygun olarak sistemimize aşağıdaki güvenlik zırhları entegre edilmiştir:

1.  **Image Quality Gate:** `Laplacian Variance` algoritması ile bulanık veya düşük çözünürlüklü görüntüler otomatik tespit edilerek "Düşük Kalite" uyarısı verilir.
2.  **Content Validation:** Standart dışı renk profiline sahip (röntgen olmayan) görüntüler için sistem uyarı üretir.
3.  **Fail-Safe Architecture:** API katmanı `try-except` blokları ve detaylı loglama (`error_log.txt`) ile koruma altına alınmıştır; sistem model hatalarında bile hizmet vermeye devam eder.
4.  **Flexible Limit:** Klinik ortamdaki yüksek çözünürlüklü veriler düşünülerek 20MB'a kadar esnek dosya desteği sunulmuştur.

## 🛠️ Teknik Altyapı
- **Backend:** FastAPI (Python)
- **Image Processing:** OpenCV
- **Model Interface:** 224x224 Input Optimization

> **⚠️ Disclaimer:** For research and demonstration purposes only. Not for clinical use.
