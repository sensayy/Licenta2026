
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time





start_time = time.time() #Incepem sa masuram timpul de antrenament pentru a putea evalua performanta modelului in termeni de timp.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        #Setam pe GPU daca este disponibil, altfel pe CPU
print(f"Using device: {device}")

# Hyper_parameters
num_epochs    = 15
batch_size    = 32
learning_rate = 0.001


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



full_dataset = torchvision.datasets.ImageFolder(root='asm_images', transform=transform_train) # Incarcam imaginile si le transformam cu transform_train, care include resize, grayscale, augmentari si normalizare
val_dataset  = torchvision.datasets.ImageFolder(root='asm_images', transform=transform_val)  
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

train_data = Subset(full_dataset, train_indices) #O metoda de a crea subseturi prin indici
val_data   = Subset(val_dataset,  val_indices)
test_data  = Subset(test_dataset, test_indices)




#Pentru ca unele clase au mult mai putin suport in antrenament (SVM de exemplu are doar 42), facem un WeightedRandomSampler ca sa le echilibram in antrenament.
targets        = [full_dataset.targets[i] for i in train_data.indices] #Luam toti indicii din datele de antrenament
class_counts   = np.bincount(targets)              # np.bincount(numere) returneaza un array in care valoarea de la indexul i reprezinta numarul de aparitii al lui i in array-ul de intrare.
class_weights  = 1.0 / class_counts                # Calculam ponderile pentru fiecare clasa ca fiind inversul numarului de aparitii al clasei respective in setul de antrenament.
sample_weights = class_weights[targets] # sample_weights este o lista care contine ponderile pentru fiecare exemplu din setul de antrenament, extrase folosind etichetele din targets.
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights)) # WeightedRandomSampler este o clasa care permite esantionarea aleatorie a datelor dintr-un dataset, folosind ponderi pentru fiecare exemplu.



train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler) 
print("Train DataLoader created successfully.") #DataLoader incarca in batch-uri pentru ca datele sunt prea mari, folosim sampler pentru antrenament unde avem nevoie, la restu nu.
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)   
test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)
print("Test DataLoader created successfully.")


def plot_confusion_matrix(loader, title):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device) 
            labels = labels.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    disp = ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=full_dataset.classes, cmap='Blues')
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig('confusion_matrix_v1_asm.png')
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))



class CNN(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        # dropout_rate este o tehnica de regularizare care ajuta la prevenirea
        # overfitting-ului, prin "oprirea" aleatorie a unui procentaj din neuroni
        super().__init__() # Calls the parent class constructor — required boilerplate for PyTorch models.

        self.base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        # Incarcam modelul EfficientNet-B3 preantrenat pe ImageNet, folosind greutatile default
        in_features = self.base.classifier[1].in_features
        # in_features reprezinta numarul de caracteristici (features) care sunt produse
        # de ultimul strat de clasificare al modelului EfficientNet-B3, inainte de a fi
        # inlocuit cu noul strat de clasificare pentru cele 9 clase din setul nostru de date.

        self.base.classifier = nn.Sequential(
            # Inlocuim stratul de clasificare al modelului EfficientNet-B3 cu un nou strat
            # care include dropout pentru regularizare si un strat linear pentru clasificare in cele 9 clase.
            nn.Dropout(dropout),               
            nn.Linear(in_features, num_classes) 
            # Strat liniar care ia numarul de caracteristici produse de ultimul strat de clasificare al modelului EfficientNet-B3 si le mapeaza la numarul de clase din setul nostru de date (9).
        )

    def forward(self, x):
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