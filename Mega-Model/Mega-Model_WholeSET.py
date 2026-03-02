import os
import numpy as np
import pandas as pd
from PIL import Image
import xgboost as xgb
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = 'csv/Full_Features_vocab2.csv'
BYTES_IMG_DIR = 'images'
ASM_IMG_DIR   = 'asm_images'
ID_COL        = 'Id'
LABEL_COL     = 'Class'
IMG_EXT       = '.png'

W_XGB      = 0.30
W_BYTE_CNN = 0.40
W_ASM_CNN  = 0.30

CLASS_NAMES = [
    'Ramnit', 'Lollipop', 'Kelihos_v3', 'Vundo', 'Simda',
    'Tracur', 'Kelihos_v1', 'Obfuscator', 'Gatak'
]


class CNN(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        super().__init__()
        self.base = models.efficientnet_b4(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)

class CNN_ASM(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        super().__init__()
        self.base = models.efficientnet_b3(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)


# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Load models ───────────────────────────────────────────────────────────────
with open('Mega-Model/xgb_malware_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

byte_cnn = CNN(num_classes=9).to(device)
byte_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v2.pth',
                                     map_location=device, weights_only=True))
byte_cnn.eval()
print("Bytes CNN loaded successfully.")

asm_cnn = CNN_ASM(num_classes=9).to(device)
asm_cnn.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v1_asm.pth',
                                    map_location=device, weights_only=True))
asm_cnn.eval()
print("ASM CNN loaded successfully.")


# ── Verify XGBoost output shape ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna()
feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]

_test_pred = xgb_model.predict_proba(df[feature_cols].iloc[:1].values)
assert _test_pred.shape == (1, 9), (
    f"XGBoost output shape is {_test_pred.shape} — expected (1, 9)."
)
print(f"XGBoost output shape OK: {_test_pred.shape}")
print(f"Loaded {len(df)} samples | {len(feature_cols)} tabular features\n")

# ── Transforms (defined once, outside loop) ───────────────────────────────────
transform_b4 = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_b3 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Inference loop ────────────────────────────────────────────────────────────
all_preds, all_probs, all_labels = [], [], []
skipped = []
fallback_counts = {'all_three': 0, 'no_bytes': 0, 'no_asm': 0, 'xgb_only': 0}

for _, row in tqdm(df.iterrows(), total=len(df)):
    file_id = row[ID_COL]
    label   = int(row[LABEL_COL]) - 1  # convert 1-9 → 0-8

    bytes_path = os.path.join(BYTES_IMG_DIR, str(int(row[LABEL_COL])), f"{file_id}{IMG_EXT}")
    asm_path   = os.path.join(ASM_IMG_DIR,   str(int(row[LABEL_COL])), f"{file_id}{IMG_EXT}")

    has_bytes = os.path.exists(bytes_path)
    has_asm   = os.path.exists(asm_path)

    try:
        # ── XGBoost ───────────────────────────────────────────────────────
        tab_feats = row[feature_cols].values.reshape(1, -1)
        p_xgb     = xgb_model.predict_proba(tab_feats)  # (1, 9)

        # ── Bytes CNN ─────────────────────────────────────────────────────
        if has_bytes:
            img_b4 = Image.open(bytes_path)
            with torch.no_grad():
                tensor_b4 = transform_b4(img_b4).unsqueeze(0).to(device)
                p_bytes   = torch.softmax(byte_cnn(tensor_b4), dim=1).cpu().numpy()
        else:
            p_bytes = None

        # ── ASM CNN ───────────────────────────────────────────────────────
        if has_asm:
            img_b3 = Image.open(asm_path)
            with torch.no_grad():
                tensor_b3 = transform_b3(img_b3).unsqueeze(0).to(device)
                p_asm     = torch.softmax(asm_cnn(tensor_b3), dim=1).cpu().numpy()
        else:
            p_asm = None

        # ── Weighted fusion ───────────────────────────────────────────────
        if has_bytes and has_asm:
            w_xgb, w_bytes, w_asm = W_XGB, W_BYTE_CNN, W_ASM_CNN
            fallback_counts['all_three'] += 1
        elif has_bytes and not has_asm:
            w_xgb, w_bytes, w_asm = W_XGB, W_BYTE_CNN, 0.0
            fallback_counts['no_asm'] += 1
        elif has_asm and not has_bytes:
            w_xgb, w_bytes, w_asm = W_XGB, 0.0, W_ASM_CNN
            fallback_counts['no_bytes'] += 1
        else:
            w_xgb, w_bytes, w_asm = 1.0, 0.0, 0.0
            fallback_counts['xgb_only'] += 1

        # Renormalize weights to sum to 1
        total  = w_xgb + w_bytes + w_asm
        w_xgb /= total; w_bytes /= total; w_asm /= total

        p_final = w_xgb * p_xgb
        if p_bytes is not None: p_final += w_bytes * p_bytes
        if p_asm   is not None: p_final += w_asm   * p_asm

        all_preds.append(int(np.argmax(p_final, axis=1)[0]))
        all_probs.append(p_final[0])
        all_labels.append(label)

    except Exception as e:
        skipped.append((file_id, str(e)))
        if len(skipped) <= 3:
            print(f"  ↳ Error on {file_id}: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nResults:")
print(f"  Evaluated : {len(all_labels)}")
print(f"  Skipped   : {len(skipped)}")
print(f"\nFallback breakdown:")
print(f"  All 3 models  : {fallback_counts['all_three']}")
print(f"  No ASM image  : {fallback_counts['no_asm']}")
print(f"  No bytes image: {fallback_counts['no_bytes']}")
print(f"  XGBoost only  : {fallback_counts['xgb_only']}")

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# ── Metrics ───────────────────────────────────────────────────────────────────
acc = accuracy_score(all_labels, all_preds)
ll  = log_loss(all_labels, all_probs)
print(f"\nOverall Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Log-Loss         : {ll:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── Confusion matrices ────────────────────────────────────────────────────────
cm            = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, data, fmt, title in zip(
    axes,
    [cm, cm_normalized],
    ['d', '.2f'],
    ['Confusion Matrix (Counts)', 'Confusion Matrix (Normalized)']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── Confidence distribution ───────────────────────────────────────────────────
max_confs = all_probs.max(axis=1)
correct   = all_preds == all_labels

print(f"\nMean confidence (correct)  : {max_confs[correct].mean():.4f}")
print(f"Mean confidence (incorrect): {max_confs[~correct].mean():.4f}")

plt.figure(figsize=(10, 4))
plt.hist(max_confs[correct],  bins=40, alpha=0.6, color='green', label='Correct')
plt.hist(max_confs[~correct], bins=40, alpha=0.6, color='red',   label='Incorrect')
plt.xlabel('Max Softmax Confidence')
plt.ylabel('Count')
plt.title('Confidence Distribution')
plt.legend()
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=150)
plt.show()