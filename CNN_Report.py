import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image

# ── Copy your class definitions in here (MalwareDataset, MalwareCNN) ──────────
# (or import them from your training script)

class MalwareDataset(Dataset): #Inherits from Dataset from torch

    def __init__(self, image_dir, labels_dict, transform=None): #Constructor
        self.image_dir = image_dir 
        self.samples = list(labels_dict.items())  # [(filename, label), ...] #lista de perechi dintre id si clasa din dictionar
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, fname + ".png")
        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

class MalwareCNN(nn.Module): #Definirea Neural Networkului

    def __init__(self, num_classes=9): 
        super().__init__() #Calls the parent class constructor — required boilerplate for PyTorch models.

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # H/2 x 128

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # H/4 x 64

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # H/8 x 32
        )

        # Collapse variable height → fixed (4x4) feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # randomly zeros 50% of neurons during training to prevent overfitting
            nn.Linear(256, num_classes) # 256 → 9 (one score per malware class)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


# ── Class names for the 9 malware families ────────────────────────────────────
CLASS_NAMES = [
    "Ramnit", "Lollipop", "Kelihos_ver3", "Vundo", "Simda",
    "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak"
]

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MalwareCNN(num_classes=9)
model.load_state_dict(torch.load("malware_cnn.pth", map_location=device))
model.to(device)
model.eval()  # important!

# ── Load validation set (same split as training) ──────────────────────────────
df = pd.read_csv("trainLabels.csv")
existing_files = set(os.path.splitext(f)[0] for f in os.listdir("images_resized/"))
labels_dict = {
    row["Id"]: row["Class"] - 1
    for _, row in df.iterrows()
    if row["Id"] in existing_files
}

from sklearn.model_selection import train_test_split
items = list(labels_dict.items())
_, val_items = train_test_split(items, test_size=0.2, random_state=42,
                                 stratify=[v for _, v in items])
val_labels = dict(val_items)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_dataset = MalwareDataset("images_resized/", val_labels, transform=transform)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# ── Run inference ─────────────────────────────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Classification Report ─────────────────────────────────────────────────────
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,          # show numbers in each cell
    fmt="d",             # integer format
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()