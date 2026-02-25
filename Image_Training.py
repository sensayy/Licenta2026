import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


from PIL import Image
import os
import numpy as np
import pandas as pd



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

transform = transforms.Compose([
    transforms.ToTensor(),           # PIL image → float tensor, scales pixel values from [0,255] to [0,1]
    transforms.Normalize((0.5,), (0.5,)) # rescales to [-1, 1]: (pixel - 0.5) / 0.5
])

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    df = pd.read_csv("trainLabels.csv") 
    existing_files = set(os.path.splitext(f)[0] for f in os.listdir("images/"))
    labels_dict = {
        row["Id"]: row["Class"] - 1
        for _, row in df.iterrows()
        if row["Id"] in existing_files
    }
    print(f"Total in CSV: {len(df)}, Found on disk: {len(labels_dict)}, Skipped: {len(df) - len(labels_dict)}")

    # Train/val split
    items = list(labels_dict.items())
    train_items, val_items = train_test_split(items, test_size=0.2, random_state=42, 
                                               stratify=[v for _, v in items])
    train_labels = dict(train_items)
    val_labels   = dict(val_items)

    train_dataset = MalwareDataset("images_resized/", train_labels, transform=transform)
    val_dataset   = MalwareDataset("images_resized/", val_labels,   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

    model     = MalwareCNN(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(20):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "malware_cnn2.pth")