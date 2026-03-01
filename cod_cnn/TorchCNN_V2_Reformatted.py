#============================================================================
#          COD REFORMATAT DE CLAUDE PENTRU CLARITATE SI STRUCTURA
#===========================================================================



# =============================================================================
# MALWARE CLASSIFICATION - EfficientNet-B4 Fine-tuning
# Microsoft BIG 2015 Malware Classification Challenge
# Grayscale byte visualization images → 9-class classifier
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time


# =============================================================================
# SETUP - Device & Hyperparameters
# =============================================================================
start_time = time.time() #Incepem sa masuram timpul de antrenament pentru a putea evalua performanta modelului in termeni de timp.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        #Setam pe GPU daca este disponibil, altfel pe CPU
print(f"Using device: {device}")

# Hyper_parameters
num_epochs    = 15
batch_size    = 32
learning_rate = 0.001


# =============================================================================
# DATA TRANSFORMS
# Datasetul are imagini cu valori intre 0 si 255, deci normalizam la intervalul
# [0, 1] si apoi la intervalul [-1, 1] pentru a ajuta la antrenarea retelei.
# =============================================================================

# Transform pentru antrenament - fara augmentari geometrice deoarece imaginile
# malware au o structura spatiala fixa (header sus, cod la mijloc, date jos)
transform_train = transforms.Compose([
    transforms.Resize((380, 380)),               # EfficientNet-B4 native resolution
    transforms.Grayscale(num_output_channels=3), # Convertim 1 channel pe 3 pentru a putea folosi EfficientNet-B4
    # Am scos flip-urile orizontale si verticale avand in vedere ca
    # imaginile malware nu au o orientare specifica
    transforms.ToTensor(),                       # Transformam in Tensor, scaland pixelii la [0, 1]
    transforms.Normalize(
        [0.485, 0.456, 0.406],                   # Valori de mean si standard deviation pentru normalizarea imaginilor,
        [0.229, 0.224, 0.225]                    # preluate din ImageNet pentru 3 canale RGB
    )
])

'''
When EfficientNet-B4 was pretrained on ImageNet,
all images were normalized with these exact statistics.
So when you fine-tune it, you need to normalize your input the same way —
otherwise the activations in the early layers will be completely off from
what the pretrained weights expect, essentially breaking the transfer learning.
'''

# Transform pentru validare si testare - fara augmentari, doar resize si normalizare
transform_val = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =============================================================================
# DATASET LOADING & SPLITTING
# Facem splitul de train/val/test: 80% train, 10% val, 10% test
# Trei instante separate ale datasetului pentru a aplica transformari diferite
# =============================================================================
full_dataset = torchvision.datasets.ImageFolder(root='images', transform=transform_train) # Incarcam imaginile si le transformam cu transform_train
print("Images loaded successfully.")
val_dataset  = torchvision.datasets.ImageFolder(root='images', transform=transform_val)   # Pentru validare si testare nu aplicam augmentari
print("Validation images loaded successfully.")
test_dataset = torchvision.datasets.ImageFolder(root='images', transform=transform_val)
print("Test images loaded successfully.")

total   = len(full_dataset)
n_train = int(0.8 * total)
n_val   = int(0.1 * total)
n_test  = total - n_train - n_val

# .tolist() converteste tensorul de indici intr-o lista, pentru a putea fi folositi in Subset.
indices       = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist() # Returns a random permutation of integers from 0 to n - 1.
train_indices = indices[:n_train]                # Primii n_train indici pentru setul de antrenament
val_indices   = indices[n_train:n_train + n_val] # Urmatorii n_val indici pentru setul de validare
test_indices  = indices[n_train + n_val:]        # Ultimii n_test indici pentru setul de testare
print(f"Dataset split: {n_train} train, {n_val} val, {n_test} test")

# Subset este o clasa care permite crearea unui subset dintr-un dataset folosind
# o lista de indici, asigurand un split aleatoriu dar reproducibil datorita
# seed-ului fixat la generator.
train_data = Subset(full_dataset, train_indices)
val_data   = Subset(val_dataset,  val_indices)
test_data  = Subset(test_dataset, test_indices)
print("Subsets created successfully.")


# =============================================================================
# CLASS IMBALANCE HANDLING - Weighted Random Sampler
# BIG2015 este dezechilibrat (ex: Ramnit ~1000 vs Simile ~42 exemple)
# =============================================================================

# targets este o lista care contine etichetele (clasele) pentru fiecare imagine
# din setul de antrenament, extrase folosind indicii din train_data.indices.
targets       = [full_dataset.targets[i] for i in train_data.indices]
class_counts  = np.bincount(targets)              # np.bincount returneaza numarul de aparitii al fiecarei clase
class_weights = 1.0 / class_counts                # Ponderile sunt inversul numarului de aparitii al fiecarei clase
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) # Esantionare aleatorie cu ponderi pentru echilibrarea claselor
print("Weighted sampler created successfully.")


# =============================================================================
# DATA LOADERS
# =============================================================================
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler) # Incarcam datele in batch-uri folosind sampler-ul pentru echilibrarea claselor
print("Train DataLoader created successfully.")
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)   # Pentru validare si testare nu amestecam datele
print("Validation DataLoader created successfully.")
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)   # asigurand o evaluare consistenta
print("Test DataLoader created successfully.")


# =============================================================================
# MODEL DEFINITION - EfficientNet-B4 with custom classifier head
# =============================================================================
class CNN(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        # dropout_rate este o tehnica de regularizare care ajuta la prevenirea
        # overfitting-ului, prin "oprirea" aleatorie a unui procentaj din neuroni
        # in timpul antrenamentului, fortand reteaua sa invete reprezentari mai
        # robuste si generalizabile.
        super().__init__() # Calls the parent class constructor — required boilerplate for PyTorch models.

        self.base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Am schimbat efficientnet_b3 cu efficientnet_b4 pentru a vedea daca putem
        # obtine o performanta mai buna, avand in vedere ca avem un set de date
        # destul de mare (peste 10.000 de imagini) si un numar de clase relativ mic (9),
        # ceea ce ar putea beneficia de un model mai complex si mai puternic.
        # Incarcam modelul EfficientNet-B4 preantrenat pe ImageNet, folosind
        # greutatile implicite (DEFAULT).

        in_features = self.base.classifier[1].in_features
        # in_features reprezinta numarul de caracteristici (features) produse de
        # ultimul strat al modelului EfficientNet-B4, inainte de a fi inlocuit
        # cu noul strat de clasificare pentru cele 9 clase.

        self.base.classifier = nn.Sequential(
            # Inlocuim stratul de clasificare al modelului EfficientNet-B4 cu un nou
            # strat care include dropout pentru regularizare si un strat linear.
            nn.Dropout(dropout),              # Aplicam dropout pentru a preveni overfitting-ul
            nn.Linear(in_features, num_classes) # Strat linear: features → 9 clase
        )

    def forward(self, x):
        # .base se refera la modelul EfficientNet-B4 modificat, care include noul
        # strat de clasificare cu dropout. In metoda forward, pur si simplu trecem
        # inputul prin modelul EfficientNet-B4 modificat.
        return self.base(x)

model = CNN(num_classes=9).to(device)
print("Model initialized and moved to device successfully.")


# =============================================================================
# LOSS, OPTIMIZER & SCHEDULER
# =============================================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# CrossEntropyLoss este o functie de pierdere utilizata pentru problemele de
# clasificare multi-clasa, care combina log-softmax si negative log-likelihood
# loss intr-o singura functie.
# label_smoothing este o tehnica de regularizare care ajuta la prevenirea
# overfitting-ului, prin "netezirea" etichetelor (labels) din setul de date.

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
# AdamW este o varianta a algoritmului de optimizare Adam, care include o
# penalizare L2 (weight decay) pentru a preveni overfitting-ul.

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# CosineAnnealingLR este o strategie de ajustare a ratei de invatare care reduce
# rata de invatare in mod cosinusoidal pe parcursul antrenamentului, ajutand la
# convergenta modelului si prevenind blocarea in minime locale.

print("Criterion, optimizer and scheduler initialized successfully.")


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
def evaluate(loader):
    # loader este un DataLoader care contine datele de validare sau testare,
    # pe care dorim sa evaluam performanta modelului.
    model.eval() # Setam modelul in modul de evaluare pentru a dezactiva dropout
                 # si batch normalization, care afecteaza performanta la evaluare.
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad(): # Dezactivam calculul gradientilor pentru a reduce consumul
                          # de memorie si a imbunatati performanta in timpul evaluarii.
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device) # Trimitem catre GPU sau CPU
            outputs = model(images)                                # Obtinem predictiile modelului
            loss    = criterion(outputs, labels)                   # Calculam pierderea

            total_loss += loss.item() * images.size(0) # Inmultim cu batch size deoarece loss.item() returneaza pierderea medie pe batch
            correct    += (outputs.argmax(1) == labels).sum().item() # Numaram predictiile corecte
            total      += images.size(0)

    return total_loss / total, 100 * correct / total # Pierderea medie si acuratetea procentuala


# =============================================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# =============================================================================
def plot_confusion_matrix(loader, title):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.classes,
                yticklabels=full_dataset.classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right') # Rotim etichetele pentru a nu se suprapune
    plt.tight_layout()
    plt.savefig('confusion_matrix_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))


# =============================================================================
# TRAINING - STAGE 1: Warmup (Frozen Backbone)
#
# This is the two-stage transfer learning strategy:
# Stage 1 — Freeze the backbone, train only the head.
# When you load a pretrained EfficientNet, the features layers already contain
# useful weights learned from ImageNet — they detect edges, textures, shapes etc.
# Your new classifier head however starts with random weights.
# If you unfreeze everything immediately, the large random gradients from the
# untrained head will flow back through the entire network and corrupt those
# carefully pretrained weights. This is sometimes called "catastrophic forgetting".
# =============================================================================

# Inghetam (freeze) toate straturile de caracteristici ale modelului EfficientNet-B4,
# astfel incat doar stratul de clasificare (head) sa fie antrenat in Stage 1.
for param in model.base.features.parameters():
    param.requires_grad = False

optimizer_warmup = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)

warmup_losses = []
for epoch in range(5):
    model.train()
    running_loss = 0.0
    print(f"[Warmup {epoch+1}/5] Starting epoch...")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_warmup.zero_grad() # Resetam gradientii la zero inainte de backpropagation
                                     # pentru a preveni acumularea gradientilor din batch-urile anterioare.
        loss = criterion(model(images), labels)
        loss.backward()              # Calculeaza gradientii doar pentru head (restul e frozen)
        optimizer_warmup.step()      # Actualizeaza greutatile head-ului
        running_loss += loss.item()

    warmup_losses.append(running_loss / len(train_loader)) # Pierderea medie pe epoca
    print(f"[Warmup {epoch+1}/5] Train Loss: {warmup_losses[-1]:.4f}")
    print(f"Time elapsed: {(time.time() - start_time)/60:.2f} minutes")


# =============================================================================
# TRAINING - STAGE 2: Fine-tuning (All Layers Unfrozen)
#
# Once the head is stable, we unfreeze all layers.
# The learning rate drops from 1e-3 to 1e-4 to allow the pretrained backbone
# weights to gently adapt to malware images rather than ImageNet images,
# without large updates that would wipe out what they already know.
# =============================================================================

# Dezghetam (unfreeze) toti parametrii modelului pentru fine-tuning
for param in model.parameters():
    param.requires_grad = True
print("All layers unfrozen for fine-tuning.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# Cream un nou optimizer AdamW cu rata de invatare mai mica (1e-4) pentru o
# adaptare mai blanda a greutatilor preantrenate la noul set de date.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

train_losses, val_losses, val_accuracies = [], [], []
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"[Fine-tune {epoch+1}/{num_epochs}] Starting epoch...")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                          # Resetam gradientii la zero
        loss = criterion(model(images), labels)        # Calculam pierderea
        loss.backward()                                # Calculam gradientii pierderii
        optimizer.step()                               # Actualizam greutatile modelului
        running_loss += loss.item()

    scheduler.step() # Actualizam rata de invatare conform CosineAnnealingLR dupa fiecare epoca
    print("Scheduler step completed.")

    avg_train = running_loss / len(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "malware_cnn_best_v2.pth")
        print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")

    train_losses.append(avg_train)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"Time elapsed: {(time.time() - start_time)/60:.2f} minutes")


# =============================================================================
# EVALUATION ON TEST SET
# Incarcam cel mai bun model salvat si evaluam pe setul de testare
# =============================================================================
model.load_state_dict(torch.load("malware_cnn_best_v2.pth", weights_only=True))
test_loss, test_acc = evaluate(test_loader)

plot_confusion_matrix(test_loader, "Confusion Matrix - Test Set V2")

print(f"Best Val Acc: {best_val_acc:.2f}%")
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")


# =============================================================================
# SAVE RESULTS TO FILE
# =============================================================================
with open('results_v2.txt', 'w') as f:
    f.write(f"Best Val Acc: {best_val_acc:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n")
    f.write(f"Total training time: {(time.time() - start_time)/60:.1f} minutes\n")
    f.write("\nTraining History:\n")
    f.write(f"{'Epoch':<10}{'Train Loss':<15}{'Val Loss':<15}{'Val Acc':<10}\n")
    f.write("-" * 50 + "\n")
    for i, (tl, vl, va) in enumerate(zip(train_losses, val_losses, val_accuracies)):
        f.write(f"{i+1:<10}{tl:<15.4f}{vl:<15.4f}{va:<10.2f}\n")
    f.write("\nClassification Report:\n")
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            all_preds.extend(model(images).argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f.write(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

print("Results saved to results_v2.txt")

# Salvam modelul antrenat pentru a putea fi folosit ulterior pentru inferenta
# sau pentru a continua antrenamentul daca este necesar.
torch.save(model.state_dict(), "malware_cnn_final_v2.pth")
print("Final model saved.")


# =============================================================================
# PLOTTING - Training Curves
# Plotam curbele de pierdere (loss) pentru antrenament si validare, precum si
# acuratetea pe setul de validare, pentru a vizualiza performanta modelului.
# =============================================================================

# Plot 1: Full training loss including warmup stage
all_train_losses = warmup_losses + train_losses # 20 points total (5 warmup + 15 finetune)
plt.figure(figsize=(10, 5))
plt.plot(all_train_losses, label='Train Loss')
plt.axvline(x=5, color='gray', linestyle='--', label='Warmup end')
plt.title('Training Loss with Warmup V2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss_with_warmup_v2.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Stage 2 train/val loss + validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss V2')
plt.plot(val_losses,   label='Val Loss V2')
plt.title('Loss Curves V2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Validation Accuracy V2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves_v2.png', dpi=150, bbox_inches='tight')
plt.show()