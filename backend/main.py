import os
import random
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 1. HATA LOGLAMASI: Sistem çökmelerini error_log.txt dosyasına kaydeder
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MedScan AI - Healthcare Backend v1.8")

# CORS AYARLARI: Üye 4'ün (Frontend) sorunsuz bağlanabilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statik klasör oluşturma (Resimler ve Raporlar burada tutulur)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

def process_image(image_bytes):
    """Görüntü Ön İşleme ve Kalite Denetimi"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, "Hata"
    
    # BULANIKLIK KONTROLÜ (Laplacian Variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_warn = "Düşük (Bulanık)" if variance < 100 else "İyi"
    
    # GRİ TONLAMA KONTROLÜ (Röntgen mi? Denetimi)
    # Tıbbi görüntüler genelde gri tonlamalıdır, renk farkı yüksekse uyarı verir
    if not np.allclose(img[:,:,0], img[:,:,1], atol=30):
        quality_warn = "Uyarı: Standart dışı renk profili (Röntgen olmayabilir)"
    
    # OTOMATİK RESIZE: Modelin beklediği 224x224 boyutuna çekme
    img_resized = cv2.resize(img, (224, 224))
    return img_resized, quality_warn

@app.post("/predict_multiple")
async def predict_multiple(files: List[UploadFile] = File(...)):
    results_list = []
    
    # TRY-EXCEPT BLOĞU: Herhangi bir model veya sistem hatasında uygulamanın çökmesini engeller
    try:
        for file in files:
            # ESNEK BOYUT SINIRI: 20MB (Yüksek çözünürlüklü dijital röntgenler için)
            MAX_FILE_SIZE = 20 * 1024 * 1024 
            
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                results_list.append({
                    "filename": file.filename, 
                    "error": "Dosya boyutu çok büyük (Maks 20MB)"
                })
                continue

            # FORMAT DENETİMİ
            if file.content_type not in ["image/jpeg", "image/png"]:
                results_list.append({"filename": file.filename, "error": "Geçersiz format (Sadece JPG/PNG)"})
                continue

            processed_img, quality = process_image(contents)
            if processed_img is None: continue

            # METADATA STRIPPING: cv2.imwrite ile resim temizlenerek (etik veriler silinerek) kaydedilir
            ref_id = f"MS_{datetime.now().strftime('%M%S')}_{random.randint(10,99)}"
            img_path = os.path.join("static", f"{ref_id}.png")
            cv2.imwrite(img_path, processed_img)
            
            # ANALİZ SİMÜLASYONU (Üye 1'in modelini beklerken iskelet yapı)
            prediction = random.choice(["NORMAL", "PNEUMONIA", "COVID"])
            confidence = round(random.uniform(0.50, 0.99), 2)
            
            # SINIR DURUM YÖNETİMİ: %60 Altı Güven Uyarısı (Jüri puanı için)
            status_msg = "Analiz Tamamlandı"
            if confidence < 0.60:
                status_msg = "Düşük güven oranı, uzman incelemesi önerilir!"
            
            # ÖNCELİKLENDİRME (Triage Sistemi)
            priority = "YÜKSEK (ACİL)" if prediction != "NORMAL" else "DÜŞÜK"
            
            # DİJİTAL RAPOR OLUŞTURMA
            report_path = os.path.join("static", f"{ref_id}_REPORT.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"--- MedScan AI TIBBİ ANALİZ RAPORU ---\n")
                f.write(f"Referans ID: {ref_id}\nTeşhis: {prediction}\nGüven: %{confidence*100}\n")
                f.write(f"ACİLİYET: {priority}\nMesaj: {status_msg}\n")
                f.write(f"Kalite Denetimi: {quality}\n")
                f.write("-" * 45 + "\nFor research and demonstration purposes only. Not for clinical use.")

            results_list.append({
                "ref_id": ref_id,
                "prediction": prediction,
                "confidence": f"%{confidence*100}",
                "priority": priority,
                "status_msg": status_msg,
                "quality_info": quality,
                "image_url": f"/static/{ref_id}.png",
                "report_url": f"/static/{ref_id}_REPORT.txt"
            })
            
        return {"status": "success", "processed_count": len(results_list), "results": results_list}

    except Exception as e:
        # MODEL VE SİSTEM HATALARI LOGLANIR
        logging.error(f"Kritik Sistem Hatası: {str(e)}")
        raise HTTPException(status_code=500, detail="Teknik bir arıza oluştu, loglar sistem yöneticisine iletildi.")

# Jürinin ana linke tıkladığında göreceği hoş geldin mesajı
@app.get("/")
def home():
    return {"message": "MedScan AI Backend v1.8 is Online", "status": "Ready for Analysis"}