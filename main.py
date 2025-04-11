import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Configuración
DATA_DIR = './starting-package/data x20'
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Cargar dataset con estructura de carpetas
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print(dataset.class_to_idx)

# Dividir el dataset en 80% entrenamiento y 20% validación
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Verifica una muestra
for images, labels in train_loader:
    print(f"Imagenes: {images.shape}, Labels: {labels}")
    break
