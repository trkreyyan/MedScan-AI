import os
import shutil
import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 1. Uygulama Tanımı (Hatanı çözen satır burası!)
app = FastAPI(
    title="MedScan AI - Healthcare Analysis API",
    description="Akciğer röntgenlerinden hastalık tespiti yapan backend sistemi.",
    version="1.0.0"
)

# 2. CORS Ayarları (Frontend'in sana bağlanabilmesi için şart)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Klasör Ayarları
# 'static' klasörü analiz sonuçlarını (Grad-CAM resimlerini) tutacak
if not os.path.exists("static"):
    os.makedirs("static")

# Statik dosyaları dış dünyaya açıyoruz (Örn: ngrok-linki.app/static/result.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def check_health():
    return {
        "status": "online", 
        "message": "MedScan AI Backend is running!",
        "version": "1.0.0"
    }

# 4. ANA TAHMİN KAPISI
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- HATA YÖNETİMİ (12 Puanlık Kriter) ---
    # Dosya türü kontrolü
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Hata: Lütfen geçerli bir resim dosyası (PNG/JPG) yükleyin."
        )

    # Resmi 'static' klasörüne kaydetme
    file_path = os.path.join("static", "current_scan.png")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --- SİMÜLASYON (Üye 1 ve 2 gelene kadar) ---
    # Frontend testi için farklı sonuçlar dönüyoruz
    results = ["Normal", "Pneumonia", "COVID-19"]
    prediction = random.choice(results)
    confidence = round(random.uniform(0.70, 0.99), 2)
    
    # Isı haritası simülasyonu (Şimdilik orijinal resmi döndürüyoruz)
    grad_cam_url = "/static/current_scan.png" 

    # --- HATA YÖNETİMİ: Düşük Güven Skoru ---
    warning_message = None
    if confidence < 0.85:
        warning_message = "Düşük güven oranı! Kesin teşhis için bir radyoloğa danışın."

    return {
        "status": "success",
        "data": {
            "prediction": prediction,
            "confidence": confidence,
            "image_url": grad_cam_url,
            "warning": warning_message
        },
        "info": "Bu bir simülasyon sonucudur. Gerçek model entegrasyonu devam etmektedir."
    }