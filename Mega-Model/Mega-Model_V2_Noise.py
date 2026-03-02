import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_PATH      = 'csv/Full_Features_vocab2.csv'
BYTES_IMG_DIR = 'images'
ASM_IMG_DIR   = 'asm_images'
ID_COL        = 'Id'
LABEL_COL     = 'Class'
IMG_EXT       = '.png'

# Guard & Fusion Weights
CONFIDENCE_THRESHOLD = 0.85 
W_XGB, W_BYTE_CNN, W_ASM_CNN = 0.30, 0.40, 0.30

CLASS_NAMES = [
    'Ramnit', 'Lollipop', 'Kelihos_v3', 'Vundo', 'Simda', 
    'Tracur', 'Kelihos_v1', 'Obfuscator', 'Gatak'
]

# ── MODEL ARCHITECTURES ───────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        super().__init__()
        self.base = models.efficientnet_b4(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    def forward(self, x): return self.base(x)

class CNN_ASM(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        super().__init__()
        self.base = models.efficientnet_b3(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    def forward(self, x): return self.base(x)

# ── DEVICE & SECURE MODEL LOADING ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Initializing Engine on: {device}")

with open('Mega-Model/xgb_malware_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

byte_cnn = CNN(num_classes=9).to(device)
byte_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v2.pth', map_location=device, weights_only=True))
byte_cnn.eval()

asm_cnn = CNN_ASM(num_classes=9).to(device)
asm_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v1_asm.pth', map_location=device, weights_only=True))
asm_cnn.eval()

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
transform_b4 = transforms.Compose([
    transforms.Resize((380, 380)), transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_b3 = transforms.Compose([
    transforms.Resize((300, 300)), transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── HELPERS ───────────────────────────────────────────────────────────────────
def find_image_path(root_dir, file_id):
    """Deep search for the image file to avoid class-folder hints."""
    for root, _, files in os.walk(root_dir):
        if f"{file_id}{IMG_EXT}" in files:
            return os.path.join(root, f"{file_id}{IMG_EXT}")
    return None

def apply_robustness_corruption(img, noise_level=0.15, blur_radius=1.5):
    """In-memory image degradation."""
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(noisy_img)
    return pil_img.filter(ImageFilter.GaussianBlur(blur_radius))

# ── ANTI-LEAKAGE DATA SPLITTING ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()
total = len(df)
n_train, n_val = int(0.8 * total), int(0.1 * total)
indices = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
test_indices = indices[n_train + n_val:] 

df_hidden = df.iloc[test_indices].copy()
feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]

# ── INFERENCE ENGINE ──────────────────────────────────────────────────────────
def run_inference(dataframe, apply_stress=False, noise=0.15, blur=1.5):
    results = []
    desc = "🔥 Stress Testing" if apply_stress else "🔍 Validating Clean"
    
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
        file_id, true_label_idx = row[ID_COL], int(row[LABEL_COL]) - 1
        bytes_path = find_image_path(BYTES_IMG_DIR, file_id)
        asm_path = find_image_path(ASM_IMG_DIR, file_id)

        try:
            # 1. XGBoost
            p_xgb = xgb_model.predict_proba(row[feature_cols].values.reshape(1, -1))

            # 2. Bytes CNN
            p_bytes = None
            if bytes_path:
                img = Image.open(bytes_path).convert('RGB')
                if apply_stress: img = apply_robustness_corruption(img, noise, blur)
                tensor = transform_b4(img).unsqueeze(0).to(device)
                with torch.no_grad(): 
                    p_bytes = torch.softmax(byte_cnn(tensor), dim=1).cpu().numpy()

            # 3. ASM CNN
            p_asm = None
            if asm_path:
                img = Image.open(asm_path).convert('RGB')
                if apply_stress: img = apply_robustness_corruption(img, noise, blur)
                tensor = transform_b3(img).unsqueeze(0).to(device)
                with torch.no_grad(): 
                    p_asm = torch.softmax(asm_cnn(tensor), dim=1).cpu().numpy()

            # 4. Fusion Logic
            cw_b, cw_a = (W_BYTE_CNN if p_bytes is not None else 0), (W_ASM_CNN if p_asm is not None else 0)
            tw = W_XGB + cw_b + cw_a
            p_final = (W_XGB/tw * p_xgb)
            if p_bytes is not None: p_final += (cw_b/tw * p_bytes)
            if p_asm is not None: p_final += (cw_a/tw * p_asm)

            results.append({'True': true_label_idx, 'Pred': int(np.argmax(p_final, axis=1)[0])})
        except Exception: continue
    return pd.DataFrame(results)

# ── RUN VALIDATION ────────────────────────────────────────────────────────────
print(f"🛡️ Hidden Set Size: {len(df_hidden)}")
clean_results = run_inference(df_hidden, apply_stress=False)
clean_acc = accuracy_score(clean_results['True'], clean_results['Pred'])

stress_results = run_inference(df_hidden, apply_stress=True)
stress_acc = accuracy_score(stress_results['True'], stress_results['Pred'])

# ── REPORTING & VISUALIZATION ─────────────────────────────────────────────────
print(f"\n" + "="*45)
print(f"✅ Clean Test Accuracy   : {clean_acc*100:.2f}%")
print(f"📉 Stress Test Accuracy  : {stress_acc*100:.2f}%")
print(f"📉 Performance Drop      : {(clean_acc - stress_acc)*100:.2f}%")
print("="*45)

# Visual Comparison of first sample
sample_id = df_hidden.iloc[0][ID_COL]
path = find_image_path(BYTES_IMG_DIR, sample_id)
if path:
    orig = Image.open(path).convert('RGB')
    noisy = apply_robustness_corruption(orig)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(orig); plt.title("Original (Clean)")
    plt.subplot(1, 2, 2); plt.imshow(noisy); plt.title("Corrupted (Stress Test)")
    plt.show()