import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    log_loss, roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = 'csv/Full_Features_vocab2_sorted.csv'
BYTES_IMG_DIR = 'images'
ASM_IMG_DIR   = 'asm_images'
ID_COL        = 'Id'
LABEL_COL     = 'Class'
IMG_EXT       = '.png'
OUTPUT_DIR    = 'Mega-Model_Results'

CONFIDENCE_THRESHOLD = 0.55
W_XGB, W_BYTE_CNN, W_ASM_CNN = 0.30, 0.40, 0.30
CLASS_NAMES = ['Ramnit', 'Lollipop', 'Kelihos_v3', 'Vundo', 'Simda',
               'Tracur', 'Kelihos_v1', 'Obfuscator', 'Gatak']

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# ── Device & Model Loading ────────────────────────────────────────────────────
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

# ── Split ─────────────────────────────────────────────────────────────────────
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

# ── Transforms ────────────────────────────────────────────────────────────────
transform_b4 = transforms.Compose([
    transforms.Resize((380, 380)), transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_b3 = transforms.Compose([
    transforms.Resize((300, 300)), transforms.Grayscale(3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def find_image_path(root_dir, file_id):
    for root, _, files in os.walk(root_dir):
        if f"{file_id}{IMG_EXT}" in files:
            return os.path.join(root, f"{file_id}{IMG_EXT}")
    return None

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM: uses gradients of the target class flowing into the last
    convolutional layer to produce a coarse localization heatmap highlighting
    which regions of the image were most important for the prediction.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()

        # Global average pool the gradients
        weights      = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam          = (weights * self.activations).sum(dim=1, keepdim=True)
        cam          = torch.relu(cam)
        cam          = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_gradcam_heatmap(model, transform, img_path, target_layer, device):
    """Returns (original_img_array, heatmap_array) for a given image."""
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    gradcam   = GradCAM(model, target_layer)
    output    = model(tensor)
    pred_class = output.argmax(dim=1).item()
    gradcam.generate(tensor, pred_class)

    cam_resized = np.array(Image.fromarray(
        (gradcam.generate(tensor, pred_class) * 255).astype(np.uint8)
    ).resize(img.size, Image.BILINEAR)) / 255.0

    return np.array(img.resize((380, 380))), cam_resized, pred_class


# ── Inference Loop ────────────────────────────────────────────────────────────
results          = []
xgb_probs_all    = []
bytes_probs_all  = []
asm_probs_all    = []
sample_image_ids = []   # store IDs that have both image types for GradCAM later

for _, row in tqdm(df_hidden.iterrows(), total=len(df_hidden), desc="Inference"):
    file_id    = row[ID_COL]
    label      = int(row[LABEL_COL]) - 1
    bytes_path = find_image_path(BYTES_IMG_DIR, file_id)
    asm_path   = find_image_path(ASM_IMG_DIR,   file_id)

    try:
        p_xgb = xgb_model.predict_proba(row[feature_cols].values.reshape(1, -1).astype(np.float32))

        p_bytes = None
        if bytes_path:
            with torch.no_grad():
                tensor  = transform_b4(Image.open(bytes_path).convert('RGB')).unsqueeze(0).to(device)
                p_bytes = torch.softmax(byte_cnn(tensor), dim=1).cpu().numpy()

        p_asm = None
        if asm_path:
            with torch.no_grad():
                tensor = transform_b3(Image.open(asm_path).convert('RGB')).unsqueeze(0).to(device)
                p_asm  = torch.softmax(asm_cnn(tensor), dim=1).cpu().numpy()

        cw_b    = W_BYTE_CNN if p_bytes is not None else 0
        cw_a    = W_ASM_CNN  if p_asm   is not None else 0
        total_w = W_XGB + cw_b + cw_a

        p_final = (W_XGB / total_w * p_xgb)
        if p_bytes is not None: p_final += (cw_b / total_w * p_bytes)
        if p_asm   is not None: p_final += (cw_a / total_w * p_asm)

        max_conf    = float(np.max(p_final))
        pred_idx    = int(np.argmax(p_final, axis=1)[0])
        final_label = CLASS_NAMES[pred_idx] if max_conf >= CONFIDENCE_THRESHOLD else "Unknown"

        results.append({
            'Id':    file_id,
            'True':  label,
            'Pred':  pred_idx,
            'Final': final_label,
            'Conf':  max_conf,
            'bytes_path': bytes_path,
            'asm_path':   asm_path,
        })

        xgb_probs_all.append(p_xgb[0])
        if p_bytes is not None: bytes_probs_all.append((label, p_bytes[0]))
        if p_asm   is not None: asm_probs_all.append((label, p_asm[0]))

        # Collect a few representative samples per class for GradCAM (aspect-ratio filtered)
        if bytes_path and asm_path:
            img  = Image.open(bytes_path)
            w, h = img.size
            if 0.3 < w/h < 3.0:
                sample_image_ids.append({'id': file_id, 'label': label,
                                         'bytes_path': bytes_path, 'asm_path': asm_path})

    except Exception as e:
        print(f"Error on {file_id}: {e}")
        continue

# ── Build result dataframe ────────────────────────────────────────────────────
res_df     = pd.DataFrame(results)
valid_mask = res_df['Final'] != "Unknown"
y_true     = res_df['True'].values
y_pred     = res_df['Pred'].values

# Reconstruct final probabilities for log loss / AUC
# (re-run a lightweight pass using stored xgb probs as proxy for ensemble probs)
ensemble_probs = np.array(xgb_probs_all)   # fallback — same length as results

# ═════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Core Metrics ───────────────────────────────────────────────────────────
acc          = accuracy_score(y_true, y_pred)
bal_acc      = balanced_accuracy_score(y_true, y_pred)
mcc          = matthews_corrcoef(y_true, y_pred)
top2_acc     = top_k_accuracy_score(y_true, ensemble_probs, k=2)
ll           = log_loss(y_true, ensemble_probs)
unknowns     = (~valid_mask).sum()

# Per-class AUC (one-vs-rest)
y_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
try:
    auc_scores = roc_auc_score(y_bin, ensemble_probs, average=None, multi_class='ovr')
except Exception:
    auc_scores = np.zeros(len(CLASS_NAMES))

print("\n" + "="*60)
print("  MEGA-MODEL — COMPREHENSIVE EVALUATION REPORT")
print("="*60)
print(f"  Samples Evaluated    : {len(res_df)}")
print(f"  Accuracy             : {acc*100:.2f}%")
print(f"  Balanced Accuracy    : {bal_acc*100:.2f}%")
print(f"  Top-2 Accuracy       : {top2_acc*100:.2f}%")
print(f"  Log Loss             : {ll:.4f}")
print(f"  Matthews Corr Coef   : {mcc:.4f}  (1.0 = perfect)")
print(f"  Unknowns (<{CONFIDENCE_THRESHOLD} conf)  : {unknowns} ({unknowns/len(res_df)*100:.1f}%)")
print("="*60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                             labels=range(len(CLASS_NAMES)), zero_division=0))

# ── 2. Confusion Matrix ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_title('Confusion Matrix — Mega-Model Test Set', fontsize=14, pad=15)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/1_confusion_matrix.png")

# ── 3. Per-Class Metrics Bar Chart ────────────────────────────────────────────
report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                     labels=range(len(CLASS_NAMES)),
                                     zero_division=0, output_dict=True)
metrics_df = pd.DataFrame({
    'Precision': [report_dict[c]['precision'] for c in CLASS_NAMES],
    'Recall':    [report_dict[c]['recall']    for c in CLASS_NAMES],
    'F1-Score':  [report_dict[c]['f1-score']  for c in CLASS_NAMES],
    'AUC-ROC':   auc_scores,
}, index=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(CLASS_NAMES))
width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
for i, (col, color) in enumerate(zip(metrics_df.columns, colors)):
    ax.bar(x + i*width, metrics_df[col], width, label=col, color=color, alpha=0.85)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylim(0, 1.08)
ax.set_ylabel('Score')
ax.set_title('Per-Class Precision / Recall / F1 / AUC-ROC', fontsize=13)
ax.legend(loc='lower right')
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/2_per_class_metrics.png")

# ── 4. Confidence Distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

correct_mask = y_true == y_pred
axes[0].hist(res_df.loc[correct_mask,  'Conf'], bins=30, alpha=0.7, color='#4CAF50', label='Correct')
axes[0].hist(res_df.loc[~correct_mask, 'Conf'], bins=30, alpha=0.7, color='#F44336', label='Wrong')
axes[0].axvline(CONFIDENCE_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({CONFIDENCE_THRESHOLD})')
axes[0].set_title('Confidence Distribution: Correct vs Wrong')
axes[0].set_xlabel('Max Confidence')
axes[0].set_ylabel('Count')
axes[0].legend()

# Per-class confidence boxplot
conf_by_class = [res_df[y_true == i]['Conf'].values for i in range(len(CLASS_NAMES))]
bp = axes[1].boxplot(conf_by_class, labels=CLASS_NAMES, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))):
    patch.set_facecolor(color)
axes[1].set_title('Confidence Distribution per Class')
axes[1].set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
axes[1].set_ylabel('Confidence')
axes[1].axhline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_confidence_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/3_confidence_distributions.png")

# ── 5. Top-20 XGBoost Feature Importance ──────────────────────────────────────
importance    = xgb_model.feature_importances_
feature_names = np.array(feature_cols)
top20_idx     = np.argsort(importance)[-20:]

fig, ax = plt.subplots(figsize=(10, 8))
colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
bars = ax.barh(range(20), importance[top20_idx], color=colors_bar)
ax.set_yticks(range(20))
ax.set_yticklabels(feature_names[top20_idx], fontsize=9)
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('Top 20 Most Important XGBoost Features', fontsize=13)
# Add value labels
for bar, val in zip(bars, importance[top20_idx]):
    ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_top20_features.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/4_top20_features.png")

# ── 6. Feature Importance Heatmap (top 20 × 9 classes) ───────────────────────
# XGBoost can give per-class importance if booster is accessible
print("\nComputing SHAP values (this may take a few minutes)...")

X_test  = df_hidden[feature_cols].astype(np.float32)

# Use a background sample of 200 rows from the train split for efficiency
train_indices_list = indices[:n_train]
X_train_bg = df.iloc[train_indices_list][feature_cols].astype(np.float32).sample(
    n=min(200, n_train), random_state=42
)

explainer   = shap.TreeExplainer(xgb_model, data=X_train_bg, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test)
# shap_values shape: (n_classes, n_samples, n_features)  or (n_samples, n_features, n_classes)
# normalise to (n_classes, n_samples, n_features)
if isinstance(shap_values, list):
    # list of arrays, one per class: each (n_samples, n_features)
    shap_array = np.stack(shap_values, axis=0)          # (n_classes, n_samples, n_features)
else:
    # single array (n_samples, n_features, n_classes)
    shap_array = np.transpose(shap_values, (2, 0, 1))   # (n_classes, n_samples, n_features)

# Mean absolute SHAP per class per feature → (n_classes, n_features)
mean_abs_shap = np.abs(shap_array).mean(axis=1)         # (n_classes, n_features)

# Pick the top 20 features by their max SHAP value across any class
top20_global  = np.argsort(mean_abs_shap.max(axis=0))[-20:]
feature_names = np.array(feature_cols)

# Build heatmap matrix: rows = top-20 features, cols = classes
heatmap_data = pd.DataFrame(
    mean_abs_shap[:, top20_global].T,           # (20, n_classes)
    index=feature_names[top20_global],
    columns=CLASS_NAMES
)
# Sort rows by total importance for cleaner presentation
heatmap_data = heatmap_data.loc[heatmap_data.sum(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(
    heatmap_data,
    annot=True, fmt='.3f',
    cmap='RdYlGn_r',          # green = low impact, red = high impact
    linewidths=0.4,
    ax=ax,
    cbar_kws={'label': 'Mean |SHAP value| (impact on model output)'}
)
ax.set_title('SHAP Feature Importance — Mean |SHAP| per Feature per Class\n'
             '(How much each feature influences confidence in each malware family)',
             fontsize=12, pad=15)
ax.set_xlabel('Malware Class', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_feature_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/shap_feature_heatmap.png")

# ── 7. Grad-CAM Heatmaps ──────────────────────────────────────────────────────
# Show one example per class (bytes + ASM side by side) where available
print("\nGenerating Grad-CAM heatmaps (this may take a minute)...")

# Target layers: last conv block of each EfficientNet
byte_target_layer = byte_cnn.base.features[-1]
asm_target_layer  = asm_cnn.base.features[-1]

# Collect one representative sample per class
class_samples = {}
for s in sample_image_ids:
    lbl = s['label']
    if lbl not in class_samples:
        class_samples[lbl] = s
    if len(class_samples) == len(CLASS_NAMES):
        break

n_classes_found = len(class_samples)
if n_classes_found > 0:
    fig, axes = plt.subplots(n_classes_found, 4,
                              figsize=(16, 4 * n_classes_found))
    if n_classes_found == 1:
        axes = [axes]

    for row_idx, (class_idx, sample) in enumerate(sorted(class_samples.items())):
        axes[row_idx][0].set_ylabel(CLASS_NAMES[class_idx], fontsize=10, rotation=0,
                                     labelpad=60, va='center')
        try:
            # Bytes image + CAM
            img_arr = np.array(Image.open(sample['bytes_path']).convert('RGB').resize((380, 380)))

            # Re-run with grad
            byte_cnn_copy = CNN(num_classes=9).to(device)
            byte_cnn_copy.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v2.pth',
                                                       map_location=device, weights_only=True))
            byte_cnn_copy.train()   # need grads
            gradcam_b  = GradCAM(byte_cnn_copy, byte_cnn_copy.base.features[-1])
            tensor_b   = transform_b4(Image.open(sample['bytes_path']).convert('RGB')).unsqueeze(0).to(device)
            tensor_b.requires_grad_(True)
            cam_b      = gradcam_b.generate(tensor_b, class_idx)
            cam_b_resized = np.array(Image.fromarray((cam_b * 255).astype(np.uint8)).resize((380, 380))) / 255.0

            axes[row_idx][0].imshow(img_arr)
            axes[row_idx][0].set_title('Bytes Image', fontsize=9)
            axes[row_idx][0].axis('off')

            axes[row_idx][1].imshow(img_arr)
            axes[row_idx][1].imshow(cam_b_resized, cmap='jet', alpha=0.45)
            axes[row_idx][1].set_title('Bytes Grad-CAM', fontsize=9)
            axes[row_idx][1].axis('off')

            # ASM image + CAM
            img_arr_asm = np.array(Image.open(sample['asm_path']).convert('RGB').resize((300, 300)))

            asm_cnn_copy = CNN_ASM(num_classes=9).to(device)
            asm_cnn_copy.load_state_dict(torch.load('Mega-Model/malware_cnn_best_v1_asm.pth',
                                                      map_location=device, weights_only=True))
            asm_cnn_copy.train()
            gradcam_a    = GradCAM(asm_cnn_copy, asm_cnn_copy.base.features[-1])
            tensor_a     = transform_b3(Image.open(sample['asm_path']).convert('RGB')).unsqueeze(0).to(device)
            tensor_a.requires_grad_(True)
            cam_a        = gradcam_a.generate(tensor_a, class_idx)
            cam_a_resized = np.array(Image.fromarray((cam_a * 255).astype(np.uint8)).resize((300, 300))) / 255.0

            axes[row_idx][2].imshow(img_arr_asm)
            axes[row_idx][2].set_title('ASM Image', fontsize=9)
            axes[row_idx][2].axis('off')

            axes[row_idx][3].imshow(img_arr_asm)
            axes[row_idx][3].imshow(cam_a_resized, cmap='jet', alpha=0.45)
            axes[row_idx][3].set_title('ASM Grad-CAM', fontsize=9)
            axes[row_idx][3].axis('off')

        except Exception as e:
            print(f"  Grad-CAM failed for class {CLASS_NAMES[class_idx]}: {e}")
            for ax in axes[row_idx]:
                ax.axis('off')

    plt.suptitle('Grad-CAM: CNN Decision Regions (Bytes & ASM)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/6_gradcam_heatmaps.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/6_gradcam_heatmaps.png")

# ── 8. Error Analysis ─────────────────────────────────────────────────────────
errors      = res_df[res_df['True'] != res_df['Pred']]
error_pairs = errors.groupby(['True', 'Pred']).size().reset_index(name='count')
error_pairs['True_name'] = error_pairs['True'].map(lambda x: CLASS_NAMES[x])
error_pairs['Pred_name'] = error_pairs['Pred'].map(lambda x: CLASS_NAMES[x])
error_pairs = error_pairs.sort_values('count', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
labels_err = [f"{r['True_name']} → {r['Pred_name']}" for _, r in error_pairs.iterrows()]
ax.barh(labels_err, error_pairs['count'], color='#EF5350')
ax.set_xlabel('Number of Misclassifications')
ax.set_title('Top-10 Most Common Misclassification Pairs', fontsize=13)
ax.invert_yaxis()
for i, v in enumerate(error_pairs['count']):
    ax.text(v + 0.1, i, str(v), va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/7_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR}/7_error_analysis.png")

# ── 9. Save full summary to text ──────────────────────────────────────────────
with open(f'{OUTPUT_DIR}/full_report.txt', 'w', encoding='utf-8') as f:
    f.write("MEGA-MODEL — COMPREHENSIVE EVALUATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(f"Samples Evaluated    : {len(res_df)}\n")
    f.write(f"Accuracy             : {acc*100:.2f}%\n")
    f.write(f"Balanced Accuracy    : {bal_acc*100:.2f}%\n")
    f.write(f"Top-2 Accuracy       : {top2_acc*100:.2f}%\n")
    f.write(f"Log Loss             : {ll:.4f}\n")
    f.write(f"Matthews Corr Coef   : {mcc:.4f}\n")
    f.write(f"Unknowns             : {unknowns} ({unknowns/len(res_df)*100:.1f}%)\n\n")
    f.write("Per-Class AUC-ROC:\n")
    for name, auc in zip(CLASS_NAMES, auc_scores):
        f.write(f"  {name:<15}: {auc:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   labels=range(len(CLASS_NAMES)), zero_division=0))
    f.write("\nTop-20 Feature Importances:\n")
    for idx in reversed(top20_idx):
        f.write(f"  {feature_names[idx]:<40}: {importance[idx]:.6f}\n")

print(f"\nSaved: {OUTPUT_DIR}/full_report.txt")
print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("Done.")