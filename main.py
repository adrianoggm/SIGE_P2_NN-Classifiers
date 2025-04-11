import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import InterpolationMode

# ====================================
# Configuración de datasets principales
# ====================================

# TODO: Cargar también el dataset de x200
# Dependiendo de una variable de entorno cambiar el dataset principal: "x20" o "x200".
# Si x20 es el principal se realizará un underscaling de las de x200 a x20 para aumentar el dataset.
# Si x200 es el principal se realizará un overscaling de las de x20 a x200 para aumentar el dataset.

# Rutas para los datasets
DATA_DIR_X20 = './starting-package/data x20'
DATA_DIR_X200 = './starting-package/data x200'

# Variable de entorno para elegir el dataset principal (por defecto "x20")
MAIN_DATASET = os.environ.get("MAIN_DATASET", "x20").lower()

# Transformación estándar para validación y mapeo de clases (por defecto resize a 224x224)
standard_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

# Transformación con augmentación para entrenamiento (para mejorar la robustez)
aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Transformación para "underscaling": reescalado a un tamaño inferior (por ejemplo, 112x112)
underscale_transform = transforms.Compose([
    transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])

# Si el dataset principal es x200 se aplicará overscaling a x20 para igualar el tamaño
if MAIN_DATASET == "x20":
    main_data_dir = DATA_DIR_X20
    secondary_data_dir = DATA_DIR_X200
    # Mantener las imágenes de x20 con la transformación estándar y augmentación
    main_transform = standard_transform
    main_aug_transform = aug_transform
    # Para el dataset secundario (x200), se aplica un underscaling para que se ajuste a la escala de x20
    secondary_transform = underscale_transform
    secondary_aug_transform = secondary_transform  # Se puede optar por aumentar también con augmentación
elif MAIN_DATASET == "x200":
    main_data_dir = DATA_DIR_X200
    secondary_data_dir = DATA_DIR_X20
    # Para x200 se mantienen las imágenes originales con la transformación estándar
    main_transform = standard_transform
    main_aug_transform = aug_transform
    # Para el dataset secundario (x20), se aplica un overscaling para adaptarlas a la escala de x200.
    # Aquí se define una transformación de overscaling (por ejemplo, se escala a 448x448)
    overscale_transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    secondary_transform = overscale_transform
    secondary_aug_transform = secondary_transform
else:
    raise ValueError("La variable de entorno MAIN_DATASET debe ser 'x20' o 'x200'.")

# ====================================
# Carga de datasets y preparación del conjunto de entrenamiento
# ====================================

# Cargar el dataset principal (usado para obtener el mapeo de clases y para validación)
main_dataset = datasets.ImageFolder(main_data_dir, transform=main_transform)
print("Mapping de clases principal:", main_dataset.class_to_idx)

# Cargar el dataset secundario y aplicar la transformación correspondiente
secondary_dataset = datasets.ImageFolder(secondary_data_dir, transform=secondary_transform)

# Crear datasets con augmentación para enriquecer el pool de entrenamiento
dataset_main_aug = datasets.ImageFolder(main_data_dir, transform=main_aug_transform)
dataset_secondary_aug = datasets.ImageFolder(secondary_data_dir, transform=secondary_aug_transform)

# Combinar ambos datasets aumentados
combined_train_dataset = ConcatDataset([dataset_main_aug, dataset_secondary_aug])

# Dividir el dataset principal para validación (80% entrenamiento, 20% validación)
train_size = int(0.8 * len(main_dataset))
val_size = len(main_dataset) - train_size
train_dataset_main, val_dataset_main = random_split(main_dataset, [train_size, val_size])

# DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset_main, batch_size=BATCH_SIZE, shuffle=False)

# ====================================
# TODO: Cargar los ficheros con descripciones y visualizar la información asociada
# Posiblemente venga el valor real de estos junto con algún tipo de descripción.
# Debemos visualizar esa información por si hay algo interesante.
# También se visualizará la distribución de los datos que tenemos.
# ====================================

# Se asume que las descripciones se encuentran en el directorio "./starting-package/descriptions"
DESC_DIR = './starting-package/descriptions'
if os.path.exists(DESC_DIR):
    print("\nCargando descripciones:")
    for filename in os.listdir(DESC_DIR):
        filepath = os.path.join(DESC_DIR, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"--- {filename} ---")
                print(content)
                print("\n")
else:
    print("\nDirectorio de descripciones no encontrado.")

# Visualización de la distribución de datos: contar imágenes por clase en el dataset principal
class_counts = {}
# main_dataset.imgs es una lista de tuplas (ruta, clase)
for _, label in main_dataset.imgs:
    class_counts[label] = class_counts.get(label, 0) + 1

print("Distribución de imágenes en el dataset principal:")
for label, count in class_counts.items():
    print(f"Clase {label}: {count} imágenes")

# ====================================
# Visualización de ejemplos y análisis del ruido
# ====================================

# Mostrar una cuadrícula de ejemplos provenientes del train_loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(12, 6))
for idx in range(8):
    plt.subplot(2, 4, idx + 1)
    img = images[idx].cpu()
    label_idx = labels[idx].item()
    npimg = img.numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(npimg, 0, 1))
    plt.title(f"Clase: {label_idx}")
    plt.axis('off')
plt.suptitle("Ejemplos de Augmentación y Escalado/Underscaling/Overscaling")
plt.show()

# Visualización de la distribución de valores de píxeles (análisis de ruido)
all_pixels = images.view(-1).numpy()

plt.figure(figsize=(8, 4))
plt.hist(all_pixels, bins=50, color='gray', alpha=0.7)
plt.title("Distribución de valores de píxeles en el batch")
plt.xlabel("Valor de píxel")
plt.ylabel("Frecuencia")
plt.show()

# En este punto se puede proceder con el entrenamiento del modelo utilizando train_loader y val_loader.
# Por ejemplo:
# for epoch in range(EPOCHS):
#     # Entrenamiento y validación...
#     pass
