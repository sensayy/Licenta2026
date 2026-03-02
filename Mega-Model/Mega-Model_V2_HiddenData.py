import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = 'csv/Full_Features_vocab2.csv'
BYTES_IMG_DIR = 'images'
ASM_IMG_DIR   = 'asm_images'
ID_COL        = 'Id'
LABEL_COL     = 'Class'
IMG_EXT       = '.png'

# Guard Configuration
CONFIDENCE_THRESHOLD = 0.85 

# Weights
W_XGB, W_BYTE_CNN, W_ASM_CNN = 0.10, 0.60, 0.30
CLASS_NAMES = ['Ramnit', 'Lollipop', 'Kelihos_v3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_v1', 'Obfuscator', 'Gatak']

# ── Model Architectures ───────────────────────────────────────────────────────
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

# ── Device & Loading ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('Mega-Model/xgb_malware_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

byte_cnn = CNN(num_classes=9).to(device)
byte_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v2.pth', map_location=device, weights_only=True))
byte_cnn.eval()

asm_cnn = CNN_ASM(num_classes=9).to(device)
asm_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v1_asm.pth', map_location=device, weights_only=True))
asm_cnn.eval()

# ── DATA SPLITTING (MATCHING TRAINING LOGIC) ──────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()

# We recreate the exact same indices used in your PyTorch training script
total = len(df)
n_train = int(0.8 * total)
n_val   = int(0.1 * total)

# USE SAME SEED (42) AS TRAINING
indices = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
test_indices = indices[n_train + n_val:] 

# THIS IS THE TRUE HIDDEN SET
df_hidden = df.iloc[test_indices].copy()
feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]

print(f"🛡️ Anti-Leakage Check: Testing on {len(df_hidden)} samples (The exact 10% Test Split)")

# ── Helpers & Transforms ──────────────────────────────────────────────────────
def find_image_path(root_dir, file_id):
    for root, dirs, files in os.walk(root_dir):
        if f"{file_id}{IMG_EXT}" in files:
            return os.path.join(root, f"{file_id}{IMG_EXT}")
    return None

transform_b4 = transforms.Compose([
    transforms.Resize((380, 380)), transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_b3 = transforms.Compose([
    transforms.Resize((300, 300)), transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Inference Loop ────────────────────────────────────────────────────────────
results = []

for _, row in tqdm(df_hidden.iterrows(), total=len(df_hidden)):
    file_id = row[ID_COL]
    label   = int(row[LABEL_COL]) - 1
    
    bytes_path = find_image_path(BYTES_IMG_DIR, file_id)
    asm_path   = find_image_path(ASM_IMG_DIR, file_id)

    try:
        # 1. XGBoost Probabilities
        p_xgb = xgb_model.predict_proba(row[feature_cols].values.reshape(1, -1))

        # 2. Bytes CNN Probabilities
        p_bytes = None
        if bytes_path:
            with torch.no_grad():
                tensor = transform_b4(Image.open(bytes_path)).unsqueeze(0).to(device)
                p_bytes = torch.softmax(byte_cnn(tensor), dim=1).cpu().numpy()

        # 3. ASM CNN Probabilities
        p_asm = None
        if asm_path:
            with torch.no_grad():
                tensor = transform_b3(Image.open(asm_path)).unsqueeze(0).to(device)
                p_asm = torch.softmax(asm_cnn(tensor), dim=1).cpu().numpy()

        # 4. Weighted Fusion
        w_xgb, w_bytes, w_asm = W_XGB, (W_BYTE_CNN if p_bytes is not None else 0), (W_ASM_CNN if p_asm is not None else 0)
        total_w = w_xgb + w_bytes + w_asm
        
        p_final = (w_xgb/total_w * p_xgb)
        if p_bytes is not None: p_final += (w_bytes/total_w * p_bytes)
        if p_asm   is not None: p_final += (w_asm/total_w * p_asm)

        max_conf = np.max(p_final)
        pred_idx = int(np.argmax(p_final, axis=1)[0])
        
        # 🛡️ Threshold Guard
        final_label = CLASS_NAMES[pred_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Unknown"
        
        results.append({'True': label, 'Pred': pred_idx, 'Final': final_label, 'Conf': max_conf})

    except Exception: continue

# ── Final Reporting ───────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
valid_mask = res_df['Final'] != "Unknown"
acc = accuracy_score(res_df['True'], res_df['Pred'])

print(f"\n✅ FINAL VERDICT:")
print(f"Raw Accuracy on Test Indices: {acc*100:.2f}%")
print(f"Unknowns flagged by Guard: {len(res_df) - valid_mask.sum()}")
print(classification_report(res_df['True'], res_df['Pred'], target_names=CLASS_NAMES))