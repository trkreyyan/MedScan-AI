
!pip install timm albumentations

import os
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
from google.colab import drive

warnings.filterwarnings('ignore')

#Bu kod ile Google Drive'a bağlanarak içerisindeki tıbbi görüntüleri (Normal, Pnömoni, COVID) taradık
#ve olası bağlantı hatalarına karşı 3 denemeli bir güvenlik önlemi ekleyerek, yapay zeka modelinin okuyabilmesi için
#pandas kütüphanesi yardımıyla bu resimlerin yollarını ve etiketlerini içeren düzenli bir veri tablosu oluşturduk.

if not os.path.exists("/content/drive"):
    print("🚀 Drive bağlantısı kuruluyor...")
    drive.mount('/content/drive')


base_path = "/content/drive/MyDrive/datasate/MedScan_Data/Final_Dataset"

def create_dataset_df(split_name):
    """Klasördeki resimleri tarar. Drive hatalarına karşı dirençlidir."""
    data = []
    split_path = os.path.join(base_path, split_name)
    class_map = {'NORMAL': 0, 'PNEUMONIA': 1, 'COVID': 2}

    if os.path.exists(split_path):
        for class_folder, tag in class_map.items():
            folder_path = os.path.join(split_path, class_folder)

            # --- Hata Tolerans Mekanizması ---
            files = []
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(folder_path):

                        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        break
                except OSError as e:
                    print(f" {class_folder} okunurken Drive hatası (Errno 5), {attempt+1}. deneme yapılıyor...")
                    time.sleep(5)

            if files:
                for img in files:
                    data.append([os.path.join(folder_path, img), tag])
                print(f"✅ {split_name}/{class_folder}: {len(files)} resim yüklendi.")
            else:
                print(f"⚠️ Uyarı: {class_folder} klasörü boş veya okunamadı.")

        return pd.DataFrame(data, columns=['image_path', 'label'])
    else:
        print(f"❌ HATA: {split_path} yolu bulunamadı!")
        return pd.DataFrame(columns=['image_path', 'label'])


print("📂 Final_Dataset üzerinden veri envanteri oluşturuluyor...\n")
train_df = create_dataset_df('train')
val_df = create_dataset_df('val')
test_df = create_dataset_df('test')


print("\n" + "="*30)
if not train_df.empty:
    print(f" BAŞARILI!")
    print(f"Eğitim (Train)     : {len(train_df)} resim")
    print(f"Doğrulama (Val)    : {len(val_df)} resim")
    print(f"Test (Test)        : {len(test_df)} resim")
    print("="*30)
    print("\n Eğitim Seti Dağılımı (Train):")
    print(train_df['label'].value_counts().sort_index().rename({0: 'Normal', 1: 'Pnömoni', 2: 'COVID'}))

#görüntülerimizi yapay zeka modelinin beklediği 224x224 boyutuna standartlaştırıyor,
#tıbbi görüntülerdeki ince detayları netleştirmek için CLAHE yöntemiyle kontrastı artırıyor,
#modelin ezber yapmasını önlemek adına sadece eğitim setine rastgele çevirme ve parlaklık değişimleri uygulayıp,
#son olarak tüm verileri PyTorch'un işleyebileceği tensör formatına dönüştürüyoruz.
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 224

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

#Görüntüleri diskten okuyup hatalı dosyaları önleyen özel bir veri sınıfı oluşturduk,
#her hastalık sınıfından en fazla 6000 örnek alarak veri setini dengeledik ve son olarak
#GPU'nun tam kapasite çalışabilmesi için bu verileri 64'erli batchler halinde modele aktaracak
#hızlı taşıyıcı mekanizmaları (DataLoader) kurduk.


class MedScanDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = cv2.imread(img_path)

        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


print("🔄 Veri seti jet hızına göre kısıtlanıyor...")
train_df_reduced = train_df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 6000), random_state=42)
).reset_index(drop=True)


train_loader = DataLoader(
    MedScanDataset(train_df_reduced, train_transform),
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    MedScanDataset(val_df, val_transform),
    batch_size=64,
    shuffle=False,
    pin_memory=True
)

print(f"✅ VERİ HAZIR! Toplam Resim: {len(train_df_reduced)}")
print(f"Toplam Adım: {len(train_loader)} ")

#EfficientNet-B4'ü rojemize dahil ettik, modelin karar veren son katmanını kendi 3 sınıflı (Normal, Pnömoni, COVID)
#teşhis yapımıza uygun olarak ezberlemeyi önleyici (Dropout) eklemelerle yeniden tasarladık ve hesaplamaların çok daha hızlı gerçekleşmesi için bu yeni modelimizi ekran kartına (GPU) taşıdık.


model = timm.create_model('efficientnet_b4', pretrained=True)
num_features = model.classifier.in_features # 1792

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("✅ EfficientNet-B4 mimarisi 3 sınıf için hazır!")

#modelin zor ve belirsiz vakalara (ince lezyonlara) daha dikkatli odaklanmasını sağlayan özel bir hata ölçüm sistemi (Focal Loss) kurduk,
#verileri hızlı eğitim için tekrar paketledik ve modelimizin kendi hatalarından ders çıkarmasını sağlayacak en verimli öğrenme algoritmalarını
#(AdamW ve GPU hızını artıran GradScaler) tanımlayarak sistemi eğitime tam hazır hale getirdik.
#daha önce sayılarını eşitlediğimiz veri setine uyumlu olması için hata hesaplama ağırlıklarını dengeledik
#ve modelin hedefe yaklaştıkça daha hassas adımlar atabilmesi için öğrenme hızını otomatik olarak ayarlayan
#yeni bir zamanlayıcı (CosineAnnealingLR) ekleyerek eğitim altyapısını kusursuzlaştırdık.



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('efficientnet_b4', pretrained=True)
num_features = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, 3)
)
model = model.to(device)
print("✅ EfficientNet-B4 mimarisi 3 sınıf için hazır!")

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return ((1 - pt)**self.gamma * ce_loss).mean()


weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float).to(device)
criterion = FocalLoss(alpha=weights, gamma=2)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
scaler = torch.amp.GradScaler('cuda')

#Yapay zeka modelimizin asıl eğitim döngüsüdür. Eğitim biter bitmez modeli test edip performansını görselleştiren detaylı bir başarı raporu
#çıkardık ve son olarak modeli Google Drive'ınıza yedekledik.

local_save_path = "/content/medscan_final_model.pt"
local_backup_path = "/content/MedScan_B2_Acil_Yedek.pt"
drive_save_path = "/content/drive/MyDrive/Colab Notebooks/MedScan_Data/medscan_final_model.pt"

num_epochs = 3
print("\n🚀 Eğitim Başlıyor...")

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

        if batch_idx % 200 == 0 and batch_idx > 0:
            torch.save(model.state_dict(), local_backup_path)


    scheduler.step()

torch.save(model.state_dict(), local_save_path)
print(f"\n✅ Eğitim bitti! Model kaydedildi.")

#RAPORLAMA
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

all_labels = ['Normal', 'Pnömoni', 'COVID-19']
print("\n--- 📝 FİNAL PERFORMANS RAPORU ---")
print(classification_report(y_true, y_pred, target_names=all_labels))

# Matris
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
plt.show()


try:
    os.makedirs(os.path.dirname(drive_save_path), exist_ok=True) # Klasör yoksa oluşturur
    shutil.copy(local_save_path, drive_save_path)
    print(f"✅ model Drive'a kopyalandı!")
except Exception as e:
    print(f"⚠️ Drive kopyalama hatası  {e}")
