import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import InterpolationMode

def get_transformations():
    """
    Define y retorna las transformaciones básicas:
      - Transformación estándar (resize a 224x224 con interpolación bilineal)
      - Transformación de augmentación (random crop, flip, etc.)
      - Transformación para underscaling (resize a 112x112)
      - Transformación para overscaling (resize a 448x448)
    """
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    underscale_transform = transforms.Compose([
        transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    overscale_transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    return standard_transform, aug_transform, underscale_transform, overscale_transform

def get_dataset_configs(main_dataset_choice=None):
    """
    Configura las rutas y transformaciones a utilizar según el dataset principal.
    Por defecto, se usa el valor de la variable de entorno MAIN_DATASET (x20 o x200).
      - Si el principal es x20, se usa la transformación estándar para x20 y se aplica un underscaling a x200.
      - Si el principal es x200, se usa la transformación estándar para x200 y se aplica un overscaling a x20.
    Retorna:
      main_data_dir, secondary_data_dir, main_transform, main_aug_transform, secondary_transform, secondary_aug_transform.
    """
    if main_dataset_choice is None:
        main_dataset_choice = os.environ.get("MAIN_DATASET", "x20").lower()

    DATA_DIR_X20 = './starting-package/data x20'
    DATA_DIR_X200 = './starting-package/data x200'
    standard_transform, aug_transform, underscale_transform, overscale_transform = get_transformations()

    if main_dataset_choice == "x20":
        main_data_dir = DATA_DIR_X20
        secondary_data_dir = DATA_DIR_X200
        main_transform = standard_transform
        main_aug_transform = aug_transform
        secondary_transform = underscale_transform
        secondary_aug_transform = secondary_transform  # Puede ampliarse con augmentación si se desea
    elif main_dataset_choice == "x200":
        main_data_dir = DATA_DIR_X200
        secondary_data_dir = DATA_DIR_X20
        main_transform = standard_transform
        main_aug_transform = aug_transform
        secondary_transform = overscale_transform
        secondary_aug_transform = secondary_transform
    else:
        raise ValueError("La variable de entorno MAIN_DATASET debe ser 'x20' o 'x200'.")

    return main_data_dir, secondary_data_dir, main_transform, main_aug_transform, secondary_transform, secondary_aug_transform

def load_datasets(main_data_dir, secondary_data_dir, main_transform, main_aug_transform, secondary_transform, secondary_aug_transform):
    """
    Carga los datasets utilizando ImageFolder y aplica las transformaciones correspondientes.
    Se generan:
      - main_dataset y secondary_dataset (usados para mapeo de clases y validación)
      - versiones con augmentación para entrenamiento (dataset_main_aug y dataset_secondary_aug)
      - combined_train_dataset, que es la concatenación de las versiones aumentadas.
    Además, se divide el dataset principal en entrenamiento y validación (80/20).
    """
    main_dataset = datasets.ImageFolder(main_data_dir, transform=main_transform)
    secondary_dataset = datasets.ImageFolder(secondary_data_dir, transform=secondary_transform)
    dataset_main_aug = datasets.ImageFolder(main_data_dir, transform=main_aug_transform)
    dataset_secondary_aug = datasets.ImageFolder(secondary_data_dir, transform=secondary_aug_transform)
    combined_train_dataset = ConcatDataset([dataset_main_aug, dataset_secondary_aug])
    
    # Dividir el main_dataset (80% entrenamiento, 20% validación)
    train_size = int(0.8 * len(main_dataset))
    val_size = len(main_dataset) - train_size
    train_dataset_main, val_dataset_main = random_split(main_dataset, [train_size, val_size])
    
    return main_dataset, secondary_dataset, combined_train_dataset, train_dataset_main, val_dataset_main

def get_dataloaders(batch_size=32, main_dataset_choice=None):
    """
    Configura y retorna los datasets y los DataLoaders.
    Retorna:
      main_dataset, secondary_dataset, combined_train_dataset, train_dataset_main, val_dataset_main, train_loader, val_loader
    """
    (main_data_dir, secondary_data_dir, 
     main_transform, main_aug_transform, 
     secondary_transform, secondary_aug_transform) = get_dataset_configs(main_dataset_choice)
    
    (main_dataset, secondary_dataset, combined_train_dataset, 
     train_dataset_main, val_dataset_main) = load_datasets(main_data_dir, secondary_data_dir, main_transform, main_aug_transform, secondary_transform, secondary_aug_transform)
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_main, batch_size=batch_size, shuffle=False)
    
    return main_dataset, secondary_dataset, combined_train_dataset, train_dataset_main, val_dataset_main, train_loader, val_loader

def load_descriptions(desc_dir='./starting-package/descriptions'):
    """
    Carga y muestra el contenido de los archivos de descripción si existen en el directorio indicado.
    """
    if os.path.exists(desc_dir):
        print("\nCargando descripciones:")
        for filename in os.listdir(desc_dir):
            filepath = os.path.join(desc_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"--- {filename} ---")
                    print(content)
                    print("\n")
    else:
        print("\nDirectorio de descripciones no encontrado.")

def print_class_distribution(dataset):
    """
    Imprime la distribución de imágenes por clase (basada en el atributo imgs del dataset).
    """
    class_counts = {}
    for _, label in dataset.imgs:
        class_counts[label] = class_counts.get(label, 0) + 1

    print("Distribución de imágenes en el dataset principal:")
    for label, count in class_counts.items():
        print(f"Clase {label}: {count} imágenes")

def visualize_examples(train_loader, num_examples=8):
    """
    Muestra una cuadrícula de ejemplos extraídos del train_loader para inspeccionar las transformaciones.
    """
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    plt.figure(figsize=(12, 6))
    # Se asume una cuadrícula de 2 filas
    cols = num_examples // 2
    for idx in range(num_examples):
        plt.subplot(2, cols, idx + 1)
        img = images[idx].cpu()
        label_idx = labels[idx].item()
        npimg = img.numpy().transpose((1, 2, 0))
        plt.imshow(np.clip(npimg, 0, 1))
        plt.title(f"Clase: {label_idx}")
        plt.axis('off')
    plt.suptitle("Ejemplos de Augmentación y Escalado/Underscaling/Overscaling")
    plt.show()

def visualize_pixel_distribution(images):
    """
    Muestra un histograma de la distribución de valores de píxeles a partir de un batch de imágenes.
    """
    all_pixels = images.view(-1).numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(all_pixels, bins=50, color='gray', alpha=0.7)
    plt.title("Distribución de valores de píxeles en el batch")
    plt.xlabel("Valor de píxel")
    plt.ylabel("Frecuencia")
    plt.show()
