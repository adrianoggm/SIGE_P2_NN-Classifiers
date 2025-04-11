import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Configuración
DATA_DIR = './starting-package/data x20'
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformaciones
# 1. Transformación estándar para validación y mapeo de clases.
standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Cargar dataset con estructura de carpetas
# 2. Transformación con augmentación para entrenamiento.
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# 3. Transformación para "underscaling": reescalado a un tamaño inferior (ej. 112×112).
underscale_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# Función auxiliar para visualizar una imagen
def imshow(img, title=None):
    npimg = img.numpy().transpose((1,2,0))
    plt.imshow(np.clip(npimg, 0, 1))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Cargar dataset original (para obtener el mapeo de clases y usarlo en validación)
dataset = datasets.ImageFolder(DATA_DIR, transform=standard_transform)
print("Mapping de clases:", dataset.class_to_idx)

# Crear dos datasets para entrenamiento:
# - dataset_aug: con augmentación para generar variaciones (imagen tamaño 224x224).
# - dataset_under: con escalado inferior (112x112).
dataset_aug = datasets.ImageFolder(DATA_DIR, transform=aug_transform)
dataset_under = datasets.ImageFolder(DATA_DIR, transform=underscale_transform)

# Combinar ambos datasets en uno solo que enriquezca el pool de entrenamiento.
combined_train_dataset = ConcatDataset([dataset_aug, dataset_under])

# Dividir el dataset original para validación (80% entrenamiento, 20% validación)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# Se utiliza el dataset original (estándar) para la validación
_, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Visualización de ejemplos ---
# Visualizar una cuadrícula de ejemplos provenientes del train_loader para inspeccionar las transformaciones y posibles "ruidos"
dataiter = iter(train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(12, 6))
for idx in range(8):
    plt.subplot(2, 4, idx + 1)
    img = images[idx].cpu()  # Imagen en tensor
    label_idx = labels[idx].item()
    # Convertir tensor a imagen (no es necesario desnormalizar pues no se aplicó normalización)
    npimg = img.numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(npimg, 0, 1))
    plt.title(f"Clase: {label_idx}")
    plt.axis('off')
plt.suptitle("Ejemplos de Augmentación y Escalado/Underscaling")
plt.show()

# --- Visualización de la distribución de valores de píxeles (análisis de ruido) ---
# A partir del batch, obtener todos los píxeles y graficar su distribución.
all_pixels = images.view(-1).numpy()

plt.figure(figsize=(8, 4))
plt.hist(all_pixels, bins=50, color='gray', alpha=0.7)
plt.title("Distribución de valores de píxeles en el batch")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia")
plt.show()

# En este punto se puede proceder con el entrenamiento del modelo utilizando train_loader y val_loader.
