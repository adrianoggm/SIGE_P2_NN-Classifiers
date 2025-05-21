import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.transforms import InterpolationMode
from config import DATA_DIR_X20, DATA_DIR_X200, BATCH_SIZE, MAIN_DATASET

def get_transformations():
    """
    - standard_transform: resize 224×224 + ToTensor
    - aug_transforms: mirror, rotación ±15° y escala zoom (0.8–1.2) cada uno por separado
    """
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # Definimos cada transformación de augment por separado
    flip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ])
    rotate_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    affine_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.8, 1.2),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
    ])

    aug_transforms = [flip_transform, rotate_transform, affine_transform]
    return standard_transform, aug_transforms


def get_dataset_configs(main_dataset_choice=None):
    """
    Configura rutas y transformaciones según el dataset principal:
      - 'x20': main en DATA_DIR_X20, secundario en DATA_DIR_X200
      - 'x200': main en DATA_DIR_X200, secundario en DATA_DIR_X20

    Retorna:
      main_data_dir, secondary_data_dir, standard_transform, aug_transforms
    """
    if main_dataset_choice is None:
        main_dataset_choice = MAIN_DATASET

    standard_transform, aug_transforms = get_transformations()

    if main_dataset_choice == "x20":
        main_data_dir = DATA_DIR_X20
        secondary_data_dir = DATA_DIR_X200
    elif main_dataset_choice == "x200":
        main_data_dir = DATA_DIR_X200
        secondary_data_dir = DATA_DIR_X20
    else:
        raise ValueError("La variable MAIN_DATASET debe ser 'x20' o 'x200'.")

    return main_data_dir, secondary_data_dir, standard_transform, aug_transforms


def load_datasets():
    """
    1) Carga el dataset principal con transform estándar
    2) Divide en train/val (80/20)
    3) Crea subconjuntos augmentados aplicando cada una de las 3 transformaciones a train
    4) Concatena:
       - train limpio + train aumentado combinado (3x)
       - solo train aumentado (3x)

    Retorna:
      full_dataset,
      train_dataset,
      val_dataset,
      train_aug_dataset,
      combined_train_dataset
    """
    main_data_dir, _secondary_data_dir, main_transform, aug_transforms = get_dataset_configs()

    # Dataset original sin augment
    full_dataset = datasets.ImageFolder(main_data_dir, transform=main_transform)

    # División reproducible en train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Índices de train para aplicar augment
    train_idx = train_dataset.indices

    # Creamos un Subset para cada transformación de augment
    aug_subsets = []
    for aug_t in aug_transforms:
        aug_full = datasets.ImageFolder(main_data_dir, transform=aug_t)
        aug_subset = Subset(aug_full, train_idx)
        aug_subsets.append(aug_subset)

    # Dataset que contiene solo las versiones augmentadas
    train_aug_dataset = ConcatDataset(aug_subsets)

    # Concatena train limpio + cada train aumentado
    combined_train_dataset = ConcatDataset([train_dataset, *aug_subsets])

    return full_dataset, train_dataset, val_dataset, train_aug_dataset, combined_train_dataset


def custom_collate(batch):
    """Stack images and labels into tensores para entrenamiento"""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def get_dataloaders():
    """
    Configura y retorna los DataLoaders para entrenamiento y validación.

    Retorna:
      train_loader, val_loader
    """
    _, train_dataset, val_dataset, _, combined_train = load_datasets()

    train_loader = DataLoader(
        combined_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate
    )
    return train_loader, val_loader