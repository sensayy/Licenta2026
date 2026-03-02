#============================================================================
#          COD REFORMATAT DE CLAUDE PENTRU CLARITATE SI STRUCTURA
#===========================================================================


# =============================================================================
# MALWARE CLASSIFICATION - EfficientNet-B3 Fine-tuning
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

# Transform pentru antrenament - include augmentari geometrice
transform_train = transforms.Compose([
    transforms.Resize((300, 300)),               # EfficientNet-B3 native resolution
    transforms.Grayscale(num_output_channels=3), # Convertim 1 channel pe 3 pentru a putea folosi EfficientNet-B3
    transforms.ToTensor(),                       # Transformam in Tensor, scaland pixelii la [0, 1]
    transforms.Normalize(
        [0.485, 0.456, 0.406],                   # Valori de mean si standard deviation pentru normalizarea imaginilor,
        [0.229, 0.224, 0.225]                    # preluate din ImageNet pentru 3 canale RGB
    )
])

'''
When EfficientNet-B3 was pretrained on ImageNet,
all images were normalized with these exact statistics.
So when you fine-tune it, you need to normalize your input the same way —
otherwise the activations in the early layers will be completely off from
what the pretrained weights expect, essentially breaking the transfer learning.
'''

# Transform pentru validare si testare - fara augmentari, doar resize si normalizare
transform_val = transforms.Compose([
    transforms.Resize((300, 300)),
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
full_dataset = torchvision.datasets.ImageFolder(root='asm_images', transform=transform_train) # Incarcam imaginile si le transformam cu transform_train, care include resize, grayscale, augmentari si normalizare
val_dataset  = torchvision.datasets.ImageFolder(root='asm_images', transform=transform_val)   # Pentru validare si testare nu aplicam augmentari, doar resize, grayscale si normalizare
test_dataset = torchvision.datasets.ImageFolder(root='asm_images', transform=transform_val)

total   = len(full_dataset)
n_train = int(0.8 * total)
n_val   = int(0.1 * total)
n_test  = total - n_train - n_val

# .tolist() converteste tensorul de indici intr-o lista, pentru a putea fi folositi in Subset.
indices       = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist() # Returns a random permutation of integers from 0 to n - 1.
train_indices = indices[:n_train]                # Primii n_train indici pentru setul de antrenament, restul pentru validare si testare, asigurand un split aleatoriu dar reproducibil datorita seed-ului fixat la generator
val_indices   = indices[n_train:n_train + n_val] # Urmatorii n_val indici pentru setul de validare, restul pentru testare
test_indices  = indices[n_train + n_val:]        # Ultimii n_test indici pentru setul de testare

# Subset este o clasa care permite crearea unui subset dintr-un dataset,
# folosind o lista de indici. Astfel, putem crea un subset pentru antrenament
# folosind indicii generati anterior, asigurand un split aleatoriu dar
# reproducibil datorita seed-ului fixat la generator.
# De asemenea, putem face acelasi lucru pentru validare si testare,
# folosind indicii corespunzatori.
train_data = Subset(full_dataset, train_indices)
val_data   = Subset(val_dataset,  val_indices)
test_data  = Subset(test_dataset, test_indices)


# =============================================================================
# CLASS IMBALANCE HANDLING - Weighted Random Sampler
# BIG2015 este dezechilibrat (ex: Ramnit ~1000 vs Simile ~42 exemple)
# =============================================================================

# targets este o lista care contine etichetele (clasele) pentru fiecare imagine
# din setul de antrenament, extrase folosind indicii din train_data.indices.
# Aceste etichete sunt necesare pentru a calcula ponderile pentru fiecare clasa
# in cazul unui dezechilibru al claselor.
targets        = [full_dataset.targets[i] for i in train_data.indices]
class_counts   = np.bincount(targets)              # np.bincount(numere) returneaza un array in care valoarea de la indexul i reprezinta numarul de aparitii al lui i in array-ul de intrare.
class_weights  = 1.0 / class_counts                # Calculam ponderile pentru fiecare clasa ca fiind inversul numarului de aparitii al clasei respective in setul de antrenament.
sample_weights = [class_weights[t] for t in targets] # sample_weights este o lista care contine ponderile pentru fiecare exemplu din setul de antrenament, extrase folosind etichetele din targets.
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights)) # WeightedRandomSampler este o clasa care permite esantionarea aleatorie a datelor dintr-un dataset, folosind ponderi pentru fiecare exemplu.


# =============================================================================
# DATA LOADERS
# =============================================================================
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler) # DataLoader este o clasa care permite incarcarea datelor in batch-uri, folosind un sampler pentru a controla modul in care sunt selectate exemplele din dataset.
print("Train DataLoader created successfully.")
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)   # Pentru validare si testare nu amestecam datele, deoarece vrem sa evaluam modelul pe un set fix de exemple, asigurand o evaluare consistenta.
print("Validation DataLoader created successfully.")
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)
print("Test DataLoader created successfully.")

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
    plt.savefig('confusion_matrix_v1_asm.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))




# =============================================================================
# MODEL DEFINITION - EfficientNet-B3 with custom classifier head
# =============================================================================
class CNN(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        # dropout_rate este o tehnica de regularizare care ajuta la prevenirea
        # overfitting-ului, prin "oprirea" aleatorie a unui procentaj din neuroni
        # in timpul antrenamentului, fortand reteaua sa invete reprezentari mai
        # robuste si generalizabile.
        super().__init__() # Calls the parent class constructor — required boilerplate for PyTorch models.

        self.base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        # Incarcam modelul EfficientNet-B3 preantrenat pe ImageNet, folosind
        # greutatile implicite (DEFAULT) care au fost antrenate pe setul de date ImageNet.

        in_features = self.base.classifier[1].in_features
        # in_features reprezinta numarul de caracteristici (features) care sunt produse
        # de ultimul strat de clasificare al modelului EfficientNet-B3, inainte de a fi
        # inlocuit cu noul strat de clasificare pentru cele 9 clase din setul nostru de date.

        self.base.classifier = nn.Sequential(
            # Inlocuim stratul de clasificare al modelului EfficientNet-B3 cu un nou strat
            # care include dropout pentru regularizare si un strat linear pentru clasificare in cele 9 clase.
            nn.Dropout(dropout),               # Aplicam dropout cu rata specificata pentru a preveni overfitting-ul, ajutand reteaua sa invete reprezentari mai robuste si generalizabile.
            nn.Linear(in_features, num_classes) # Strat linear care ia numarul de caracteristici produse de ultimul strat de clasificare al modelului EfficientNet-B3 si le mapeaza la numarul de clase din setul nostru de date (9).
        )

    def forward(self, x):
        # .base se refera la modelul EfficientNet-B3 modificat, care include noul
        # strat de clasificare cu dropout. In metoda forward, pur si simplu trecem
        # inputul prin modelul EfficientNet-B3 modificat. Acest lucru permite modelului
        # sa invete sa clasifice imaginile in cele 9 clase din setul nostru de date,
        # folosind reprezentarile invatate de EfficientNet-B3.
        return self.base(x)

model = CNN(num_classes=9).to(device)


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
# penalizare L2 (weight decay) pentru a preveni overfitting-ul,
# ajutand reteaua sa invete reprezentari mai robuste si generalizabile.

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# CosineAnnealingLR este o strategie de ajustare a ratei de invatare
# care reduce rata de invatare in mod cosinusoidal pe parcursul antrenamentului,
# ajutand la convergenta modelului si prevenind blocarea in minime locale.


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
def evaluate(loader):
    # loader este un DataLoader care contine datele de validare sau testare,
    # pe care dorim sa evaluam performanta modelului.
    model.eval() # Setam modelul in modul de evaluare pentru a dezactiva anumite comportamente
                 # specifice antrenamentului, cum ar fi dropout-ul si batch normalization,
                 # care pot afecta performanta modelului in timpul evaluarii.
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad(): # torch.no_grad() este un context manager care dezactiveaza calculul
                          # gradientilor, ceea ce reduce consumul de memorie si imbunatateste
                          # performanta in timpul evaluarii modelului, deoarece nu avem nevoie
                          # de gradienti pentru a face predictii sau a calcula pierderea.
        for images, labels in loader: # Iteram prin batch-urile de imagini si etichete din DataLoader-ul de validare sau testare
            images, labels = images.to(device), labels.to(device) # Trimitem catre GPU sau CPU in functie de care este valabil, pentru a accelera procesul de evaluare.
            outputs = model(images)                                # Trecem imaginile prin model pentru a obtine predictiile (outputs) ale modelului, care sunt scorurile pentru fiecare clasa pentru fiecare imagine din batch.
            loss    = criterion(outputs, labels)                   # Calculam pierderea (loss) intre predictiile modelului (outputs) si etichetele reale (labels) folosind functia de pierdere definita anterior (CrossEntropyLoss cu label smoothing).

            total_loss += loss.item() * images.size(0)           # Inmultim pierderea pentru a obtine pierderea totala pe batch, deoarece loss.item() returneaza pierderea medie pe batch.
            correct    += (outputs.argmax(1) == labels).sum().item() # Calculam numarul de predictii corecte in batch, comparand indexul cu cea mai mare valoare din output (predictia modelului) cu etichetele reale (labels).
            total      += images.size(0)                         # Adaugam numarul de exemple din batch la total pentru a putea calcula acuratetea la final.

    return total_loss / total, 100 * correct / total # Returnam pierderea medie pe batch si acuratetea procentuala a modelului pe setul de date evaluat.


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
# carefully pretrained weights during the first few updates.
# This is sometimes called "catastrophic forgetting".
# This means only the new classifier head gets updated for the first 5 epochs —
# it learns to make reasonable predictions using the pretrained features
# without destroying them.
# =============================================================================

# Iteram prin toate straturile de caracteristici (features) ale modelului EfficientNet-B3,
# care sunt stocate in model.base.features, si setam requires_grad la False pentru fiecare parametru.
# In prima etapa, inghetam (freeze) toate straturile de caracteristici (features) ale modelului
# EfficientNet-B3, astfel incat doar stratul de clasificare (head) sa fie antrenat.
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
    print(f"Beginning warmup Epoch [{epoch+1}/5]")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_warmup.zero_grad() # Resetam gradientii la zero inainte de a face backpropagation,
                                     # pentru a preveni acumularea gradientilor de la batch-urile anterioare,
                                     # ceea ce ar putea duce la actualizari incorecte ale greutatilor modelului.
        loss = criterion(model(images), labels)
        loss.backward()              # .backward() calculeaza gradientii pierderii (loss) fata de toti parametrii
                                     # modelului care au requires_grad=True, adica in acest caz doar pentru stratul
                                     # de clasificare (head), deoarece restul straturilor sunt inghetate (freeze).
        optimizer_warmup.step()      # optimizer.step() actualizeaza greutatile modelului folosind gradientii calculati in pasul anterior (loss.backward())
        running_loss += loss.item()  # Adaugam pierderea curenta la running_loss pentru a putea calcula pierderea medie pe batch la finalul epocii.

    warmup_losses.append(running_loss / len(train_loader)) # Adaugam pierderea medie pe epoca la lista de pierderi pentru etapa de warmup
    print(f"[Warmup {epoch+1}/5] Train Loss: {warmup_losses[-1]:.4f}")


# =============================================================================
# TRAINING - STAGE 2: Fine-tuning (All Layers Unfrozen)
#
# Stage 2 — Unfreeze everything with a lower learning rate.
# Once the head is stable, you unfreeze all layers.
# And crucially the learning rate drops from 1e-3 to 1e-4. This allows the
# pretrained backbone weights to gently adapt to malware images rather than
# ImageNet images, without large updates that would wipe out what they already know.
# =============================================================================

# Iteram prin toti parametrii modelului, inclusiv straturile de caracteristici (features)
# si stratul de clasificare (head), si setam requires_grad la True pentru fiecare parametru.
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# Dupa ce am decongelat toate straturile modelului, cream un nou optimizer AdamW cu o
# rata de invatare mai mica (1e-4) pentru a permite o adaptare mai blanda a greutatilor
# preantrenate la noul set de date, fara a le corupe brusc.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

train_losses, val_losses, val_accuracies = [], [], []
best_val_acc = 0.0

for epoch in range(num_epochs): # Incepem antrenarea propriu-zisa pentru num_epochs (10 epoci)
    model.train()                # Setam modelul in modul antrenament
    running_loss = 0.0           # Tinem cont de pierderea cumulativa pe epoca pentru a putea calcula pierderea medie pe batch la finalul epocii.
    print(f"Beginning fine-tuning Epoch [{epoch+1}/{num_epochs}]")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                           # Resetam gradientii la zero
        loss = criterion(model(images), labels)         # Calculam pierderea intre predictiile modelului si etichetele reale folosind functia de pierdere definita anterior (CrossEntropyLoss cu label smoothing).
        loss.backward()                                 # Calculam gradientii pierderii
        optimizer.step()                                # Actualizam greutatile modelului folosind gradientii calculati
        running_loss += loss.item()                     # Adaugam pierderea curenta la running_loss pentru a putea calcula pierderea medie pe batch la finalul epocii.

    scheduler.step() # Actualizam rata de invatare conform strategiei de ajustare a ratei de invatare (CosineAnnealingLR) dupa fiecare epoca.

    print("Updated learning rate:", scheduler.get_last_lr()[0]) # Afisam rata de invatare actualizata pentru a putea monitoriza cum evolueaza pe parcursul antrenamentului.
    avg_train = running_loss / len(train_loader)
    val_loss, val_acc = evaluate(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "malware_cnn_best_v1_asm.pth")

    train_losses.append(avg_train)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


# =============================================================================
# EVALUATION ON TEST SET
# Incarcam cel mai bun model salvat si evaluam pe setul de testare
# =============================================================================
model.load_state_dict(torch.load("malware_cnn_best_v1_asm.pth", weights_only=True))
test_loss, test_acc = evaluate(test_loader)

print(f"Best Val Acc: {best_val_acc:.2f}%")
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

plot_confusion_matrix(test_loader, "Confusion Matrix - Test Set V2")

print(f"Best Val Acc: {best_val_acc:.2f}%")
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")


# Salvam modelul antrenat pentru a putea fi folosit ulterior pentru inferenta
# sau pentru a continua antrenamentul daca este necesar.
torch.save(model.state_dict(), "malware_cnn_final_v1_asm.pth")

with open('results_v1_asm.txt', 'w') as f:
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
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

print("Results saved to results_v1_asm.txt")

# =============================================================================
# PLOTTING - Training Curves
# Plotam curbele de pierdere (loss) pentru antrenament si validare, precum si
# acuratetea pe setul de validare, pentru a vizualiza performanta modelului
# pe parcursul antrenamentului.
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