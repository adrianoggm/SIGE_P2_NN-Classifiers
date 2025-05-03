import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import InterpolationMode
from config import DATA_DIR_X20, DATA_DIR_X200, BATCH_SIZE, MAIN_DATASET

def get_transformations():
    """
    - standard_transform: resize 224×224 + ToTensor
    - aug_transform: random mirror, rotación ±15° o escala zoom (0.8–1.2) + ToTensor
    """
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    aug_transform = transforms.Compose([
        # mantenemos siempre 224×224
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),

        # elegimos ALEATORIAMENTE UNA de estas 3 operaciones:
        transforms.RandomChoice([
            # 1) Espejo vertical puro
            transforms.RandomHorizontalFlip(p=1.0),

            # 2) Rotación aleatoria entre -15° y +15°
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),

            # 3) Escalado (zoom in/out) centrado,
            #    scale <1 → zoom out (se ve más contexto),
            #    scale >1 → zoom in (se ve detalle)
            transforms.RandomAffine(
                degrees=0,               # sin rotación aquí
                scale=(0.8, 1.2),        # rango de escalado
                interpolation=InterpolationMode.BILINEAR
            ),
        ]),

        transforms.ToTensor(),
    ])

    return standard_transform, aug_transform

def get_dataset_configs(main_dataset_choice=None):
    """
    Configura las rutas y transformaciones a utilizar según el dataset principal.
    - Si el principal es x20: se usa la transformación estándar para x20 y se aplica un underscaling a x200.
    - Si el principal es x200: se usa la transformación estándar para x200 y se aplica un overscaling a x20.
    
    Retorna:
      main_data_dir, secondary_data_dir, main_transform, main_aug_transform, secondary_transform, secondary_aug_transform
    """
    if main_dataset_choice is None:
        main_dataset_choice = MAIN_DATASET

    standard_transform, aug_transform = get_transformations()

    if main_dataset_choice == "x20":
        main_data_dir = DATA_DIR_X20
        secondary_data_dir = DATA_DIR_X200
        main_transform = standard_transform
        main_aug_transform = aug_transform
    elif main_dataset_choice == "x200":
        main_data_dir = DATA_DIR_X200
        secondary_data_dir = DATA_DIR_X20
        main_transform = standard_transform
        main_aug_transform = aug_transform
    else:
        raise ValueError("La variable MAIN_DATASET debe ser 'x20' o 'x200'.")

    return main_data_dir, secondary_data_dir, main_transform, main_aug_transform

def load_datasets():
    """
    Carga los datasets utilizando ImageFolder y aplica las transformaciones.
    Genera:
      - main_dataset y secondary_dataset (para mapeo y validación)
      - dataset_main_aug y dataset_secondary_aug (con transformaciones de augmentación)
      - combined_train_dataset: la concatenación de las versiones aumentadas
    Además, divide el main_dataset en entrenamiento y validación (80/20).
    """
    (main_data_dir, secondary_data_dir,
     main_transform, main_aug_transform) = get_dataset_configs()

    main_dataset = datasets.ImageFolder(main_data_dir, transform=main_transform)
    dataset_main_aug = datasets.ImageFolder(main_data_dir, transform=main_aug_transform)
    
    train_size = int(0.8 * len(main_dataset))
    val_size = len(main_dataset) - train_size
    train_dataset_main, val_dataset_main = random_split(main_dataset, [train_size, val_size])
    
    return main_dataset, train_dataset_main, val_dataset_main

def custom_collate(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels  # Devuelve listas en lugar de tensores apilados


def get_dataloaders():
    """
    Configura y retorna los DataLoaders para entrenamiento y validación.
    Retorna:
      main_dataset, secondary_dataset, combined_train_dataset,
      train_dataset_main, val_dataset_main, train_loader, val_loader
    """
    main_dataset, train_dataset_main, val_dataset_main = load_datasets()
    train_loader = DataLoader(train_dataset_main, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset_main, batch_size=BATCH_SIZE, shuffle=False)
    return main_dataset, train_dataset_main, val_dataset_main, train_loader, val_loader
