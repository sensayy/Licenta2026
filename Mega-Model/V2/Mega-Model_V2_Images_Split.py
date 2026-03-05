import os
import numpy as np
import pandas as pd
from PIL import Image
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
print(f"🚀 Initializing Inference Engine on: {device}")

# Load XGBoost (Standard Pickle)
with open('Mega-Model/xgb_malware_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Load PyTorch models with weights_only=True for security
byte_cnn = CNN(num_classes=9).to(device)
byte_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v2.pth', 
                                    map_location=device, weights_only=True))
byte_cnn.eval()

asm_cnn = CNN_ASM(num_classes=9).to(device)
asm_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v1_asm.pth', 
                                   map_location=device, weights_only=True))
asm_cnn.eval()

print("✅ Models loaded securely (weights_only=True).")

# ── ANTI-LEAKAGE DATA SPLITTING ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()
total = len(df)
n_train = int(0.8 * total)
n_val   = int(0.1 * total)

# Exact same seed and logic as the training script
indices = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
test_indices = indices[n_train + n_val:] 

df_hidden = df.iloc[test_indices].copy()
feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]

print(f"🛡️ Testing on {len(df_hidden)} samples from the VALIDATED test split.")

# ── HELPERS & TRANSFORMS ──────────────────────────────────────────────────────
def find_image_path(root_dir, file_id):
    for root, _, files in os.walk(root_dir):
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

# ── INFERENCE LOOP ────────────────────────────────────────────────────────────
results = []

for _, row in tqdm(df_hidden.iterrows(), total=len(df_hidden)):
    file_id = row[ID_COL]
    true_label_idx = int(row[LABEL_COL]) - 1
    
    bytes_path = find_image_path(BYTES_IMG_DIR, file_id)
    asm_path   = find_image_path(ASM_IMG_DIR, file_id)

    try:
        # 1. XGBoost
        p_xgb = xgb_model.predict_proba(row[feature_cols].values.reshape(1, -1))

        # 2. Bytes CNN
        p_bytes = None
        if bytes_path:
            with torch.no_grad():
                img = Image.open(bytes_path).convert('RGB') # Ensure RGB for Normalize
                tensor = transform_b4(img).unsqueeze(0).to(device)
                p_bytes = torch.softmax(byte_cnn(tensor), dim=1).cpu().numpy()

        # 3. ASM CNN
        p_asm = None
        if asm_path:
            with torch.no_grad():
                img = Image.open(asm_path).convert('RGB')
                tensor = transform_b3(img).unsqueeze(0).to(device)
                p_asm = torch.softmax(asm_cnn(tensor), dim=1).cpu().numpy()

        # 4. Adaptive Weighted Fusion
        # If an image is missing, its weight is distributed to the others
        current_w_bytes = W_BYTE_CNN if p_bytes is not None else 0
        current_w_asm   = W_ASM_CNN if p_asm is not None else 0
        total_w = W_XGB + current_w_bytes + current_w_asm
        
        p_final = (W_XGB/total_w * p_xgb)
        if p_bytes is not None: p_final += (current_w_bytes/total_w * p_bytes)
        if p_asm   is not None: p_final += (current_w_asm/total_w * p_asm)

        max_conf = np.max(p_final)
        pred_idx = int(np.argmax(p_final, axis=1)[0])
        
        # 🛡️ Apply Confidence Guard
        is_trusted = max_conf >= CONFIDENCE_THRESHOLD
        final_verdict = CLASS_NAMES[pred_idx] if is_trusted else "Unknown/Suspicious"
        
        results.append({
            'Id': file_id,
            'True': true_label_idx,
            'Pred': pred_idx,
            'Conf': max_conf,
            'Verdict': final_verdict,
            'Trusted': is_trusted
        })

    except Exception as e:
        print(f"Error processing {file_id}: {e}")
        continue

# ── FINAL PERFORMANCE ANALYSIS ────────────────────────────────────────────────
res_df = pd.DataFrame(results)
acc = accuracy_score(res_df['True'], res_df['Pred'])
unknown_count = len(res_df) - res_df['Trusted'].sum()

print(f"\n" + "="*40)
print(f"✅ FINAL VERDICT REPORT")
print(f"="*40)
print(f"Total Samples Tested : {len(res_df)}")
print(f"Raw Accuracy         : {acc*100:.2f}%")
print(f"Unknowns (Guard)     : {unknown_count} ({ (unknown_count/len(res_df))*100 :.2f}%)")
print("-" * 40)
print(classification_report(res_df['True'], res_df['Pred'], target_names=CLASS_NAMES))

# ── CONFUSION MATRIX VISUALIZATION ────────────────────────────────────────────
cm = confusion_matrix(res_df['True'], res_df['Pred'])
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Final Ensemble Confusion Matrix\n(Accuracy: {acc*100:.2f}%)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()