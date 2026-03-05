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
import shap

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_PATH      = 'csv/Full_Features_vocab2_sorted.csv'  # SORTED - Facut sa fie ca folderele cu imagini pt asm si bytes impotriva data leakage.
BYTES_IMG_DIR = 'images'
ASM_IMG_DIR   = 'asm_images'
ID_COL        = 'Id'
LABEL_COL     = 'Class'
IMG_EXT       = '.png'

CONFIDENCE_THRESHOLD = 0.55
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

# ── DEVICE & MODEL LOADING ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    for root, _, files in os.walk(root_dir):
        if f"{file_id}{IMG_EXT}" in files:
            return os.path.join(root, f"{file_id}{IMG_EXT}")
    return None

def apply_robustness_corruption(img, noise_level=0.15, blur_radius=1.5):
    """Gaussian noise + blur to simulate degraded/obfuscated malware images."""
    img_array = np.array(img).astype(np.float32)
    noise     = np.random.normal(0, noise_level * 255, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img).filter(ImageFilter.GaussianBlur(blur_radius))

# ── SPLIT IDENTIC CU CNN-URILE SI XGBOOST ────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna()
total   = len(df)
n_train = int(0.8 * total)
n_val   = int(0.1 * total)

indices      = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
test_indices = indices[n_train + n_val:]

df_hidden    = df.iloc[test_indices].reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in [ID_COL, LABEL_COL]]

print(f"Total dataset:  {total}")
print(f"Test set size:  {len(df_hidden)} samples (10%)")

# ── INFERENCE ENGINE ──────────────────────────────────────────────────────────
def run_inference(dataframe, apply_stress=False, noise=0.3, blur=2.5):
    results = []
    desc = "Stress Testing" if apply_stress else "Validating Clean"

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
        file_id        = row[ID_COL]
        true_label_idx = int(row[LABEL_COL]) - 1
        bytes_path     = find_image_path(BYTES_IMG_DIR, file_id)
        asm_path       = find_image_path(ASM_IMG_DIR, file_id)

        try:
            # 1. XGBoost
            p_xgb = xgb_model.predict_proba(row[feature_cols].values.reshape(1, -1).astype(np.float32))

            # 2. Bytes CNN
            p_bytes = None
            if bytes_path:
                img = Image.open(bytes_path).convert('RGB')
                if apply_stress:
                    img = apply_robustness_corruption(img, noise, blur)
                with torch.no_grad():
                    tensor  = transform_b4(img).unsqueeze(0).to(device)
                    p_bytes = torch.softmax(byte_cnn(tensor), dim=1).cpu().numpy()

            # 3. ASM CNN
            p_asm = None
            if asm_path:
                img = Image.open(asm_path).convert('RGB')
                if apply_stress:
                    img = apply_robustness_corruption(img, noise, blur)
                with torch.no_grad():
                    tensor = transform_b3(img).unsqueeze(0).to(device)
                    p_asm  = torch.softmax(asm_cnn(tensor), dim=1).cpu().numpy()

            # 4. Weighted fusion
            cw_b = W_BYTE_CNN if p_bytes is not None else 0
            cw_a = W_ASM_CNN  if p_asm   is not None else 0
            tw   = W_XGB + cw_b + cw_a

            p_final = (W_XGB / tw * p_xgb)
            if p_bytes is not None: p_final += (cw_b / tw * p_bytes)
            if p_asm   is not None: p_final += (cw_a / tw * p_asm)

            results.append({
                'True': true_label_idx,
                'Pred': int(np.argmax(p_final, axis=1)[0]),
                'Conf': float(np.max(p_final))
            })

        except Exception as e:
            print(f"Error on {file_id}: {e}")
            continue

    return pd.DataFrame(results)

# ── RUN BOTH PASSES ───────────────────────────────────────────────────────────
clean_results  = run_inference(df_hidden, apply_stress=False)
stress_results = run_inference(df_hidden, apply_stress=True)

clean_acc  = accuracy_score(clean_results['True'],  clean_results['Pred'])
stress_acc = accuracy_score(stress_results['True'], stress_results['Pred'])

# ── REPORTING ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Clean Test Accuracy :  {clean_acc*100:.2f}%")
print(f"Stress Test Accuracy:  {stress_acc*100:.2f}%")
print(f"Performance Drop    :  {(clean_acc - stress_acc)*100:.2f}%")
print(f"{'='*50}")

print("\nClean Classification Report:")
print(classification_report(
    clean_results['True'], clean_results['Pred'],
    target_names=CLASS_NAMES, labels=range(len(CLASS_NAMES)), zero_division=0
))

print("\nStress Classification Report:")
print(classification_report(
    stress_results['True'], stress_results['Pred'],
    target_names=CLASS_NAMES, labels=range(len(CLASS_NAMES)), zero_division=0
))

# ── CONFUSION MATRICES ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, results, title in zip(
    axes,
    [clean_results, stress_results],
    ['Confusion Matrix - Clean', 'Confusion Matrix - Stress Test']
):
    cm = confusion_matrix(results['True'], results['Pred'], labels=range(len(CLASS_NAMES)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Mega-Model/stress_test_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── VISUAL SAMPLE COMPARISON ──────────────────────────────────────────────────
sample_img, sample_path = None, None
for _, row in df_hidden.iterrows():
    path = find_image_path(BYTES_IMG_DIR, row[ID_COL])
    if path:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if 0.5 < w/h < 2.0:  # roughly square-ish
            sample_img  = img
            sample_path = path
            break

if sample_img:
    noisy = apply_robustness_corruption(sample_img)  # exaggerated for visibility
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(sample_img); plt.title("Original (Clean)");        plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(noisy);       plt.title("Corrupted (Stress Test)"); plt.axis('off')
    plt.tight_layout()
    plt.savefig('Mega-Model/stress_test_sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()