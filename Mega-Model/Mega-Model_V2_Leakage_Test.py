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
from sklearn.model_selection import train_test_split
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = 'csv/Full_Features_vocab2.csv'
BYTES_IMG_DIR = 'images'
ASM_IMG_DIR   = 'asm_images'
ID_COL        = 'Id'
LABEL_COL     = 'Class'
IMG_EXT       = '.png'

CONFIDENCE_THRESHOLD = 0.85 
W_XGB, W_BYTE_CNN, W_ASM_CNN = 0.30, 0.40, 0.30
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

# ── 🛡️ LEAK-PROOF DATA SPLITTING ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()

# 1. Identify what PyTorch considered "Test" (The last 10%)
total = len(df)
n_train, n_val = int(0.8 * total), int(0.1 * total)
torch_indices = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
pytorch_test_ids = set(df.iloc[torch_indices[n_train + n_val:]][ID_COL])

# 2. Identify what Sklearn considered "Test" (Assuming standard 20% split used for XGB)
_, df_test_sklearn = train_test_split(df, test_size=0.20, random_state=42, stratify=df[LABEL_COL])
sklearn_test_ids = set(df_test_sklearn[ID_COL])

# 3. Find the Intersect (Samples that are UNSEEN by both model types)
clean_test_ids = pytorch_test_ids.intersection(sklearn_test_ids)

# Filter the dataframe to only these strictly unseen samples
df_hidden = df[df[ID_COL].isin(clean_test_ids)].copy()
feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]



print(f"📊 Total Dataset: {total}")
print(f"🔬 PyTorch Test Candidates: {len(pytorch_test_ids)}")
print(f"🔬 Sklearn Test Candidates: {len(sklearn_test_ids)}")
print(f"🛡️ TRUE HIDDEN SAMPLES (Zero-Leakage): {len(df_hidden)}")

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
        # 1. XGBoost
        p_xgb = xgb_model.predict_proba(row[feature_cols].values.reshape(1, -1))

        # 2. Bytes CNN
        p_bytes = None
        if bytes_path:
            with torch.no_grad():
                tensor = transform_b4(Image.open(bytes_path).convert('RGB')).unsqueeze(0).to(device)
                p_bytes = torch.softmax(byte_cnn(tensor), dim=1).cpu().numpy()

        # 3. ASM CNN
        p_asm = None
        if asm_path:
            with torch.no_grad():
                tensor = transform_b3(Image.open(asm_path).convert('RGB')).unsqueeze(0).to(device)
                p_asm = torch.softmax(asm_cnn(tensor), dim=1).cpu().numpy()

        # 4. Weighted Fusion logic
        cw_b = W_BYTE_CNN if p_bytes is not None else 0
        cw_a = W_ASM_CNN if p_asm is not None else 0
        total_w = W_XGB + cw_b + cw_a
        
        p_final = (W_XGB/total_w * p_xgb)
        if p_bytes is not None: p_final += (cw_b/total_w * p_bytes)
        if p_asm   is not None: p_final += (cw_a/total_w * p_asm)

        max_conf = np.max(p_final)
        pred_idx = int(np.argmax(p_final, axis=1)[0])
        
        final_label = CLASS_NAMES[pred_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Unknown"
        results.append({'True': label, 'Pred': pred_idx, 'Final': final_label, 'Conf': max_conf})

    except Exception: continue

# ── Final Reporting ───────────────────────────────────────────────────────────
# ── Updated Final Reporting ───────────────────────────────────────────────────
res_df = pd.DataFrame(results)
valid_mask = res_df['Final'] != "Unknown"
acc = accuracy_score(res_df['True'], res_df['Pred'])

print(f"\n✅ LEAK-PROOF FINAL VERDICT:")
print(f"Samples Evaluated: {len(res_df)}")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Unknowns (Below {CONFIDENCE_THRESHOLD}): {len(res_df) - valid_mask.sum()}")
print("-" * 30)

# The Fix: Added 'labels=range(len(CLASS_NAMES))'
print(classification_report(
    res_df['True'], 
    res_df['Pred'], 
    target_names=CLASS_NAMES, 
    labels=range(len(CLASS_NAMES)),
    zero_division=0
))