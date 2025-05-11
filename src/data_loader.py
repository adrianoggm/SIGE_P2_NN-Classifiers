import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
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
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=0,
                scale=(0.8, 1.2),
                interpolation=InterpolationMode.BILINEAR
            ),
        ]),
        transforms.ToTensor(),
    ])

    return standard_transform, aug_transform


def get_dataset_configs(main_dataset_choice=None):
    """
    Configura rutas y transformaciones según el dataset principal:
      - 'x20': main en DATA_DIR_X20, secundario en DATA_DIR_X200
      - 'x200': main en DATA_DIR_X200, secundario en DATA_DIR_X20

    Retorna:
      main_data_dir, secondary_data_dir, main_transform, main_aug_transform
    """
    if main_dataset_choice is None:
        main_dataset_choice = MAIN_DATASET

    standard_transform, aug_transform = get_transformations()

    if main_dataset_choice == "x20":
        main_data_dir = DATA_DIR_X20
        secondary_data_dir = DATA_DIR_X200
    elif main_dataset_choice == "x200":
        main_data_dir = DATA_DIR_X200
        secondary_data_dir = DATA_DIR_X20
    else:
        raise ValueError("La variable MAIN_DATASET debe ser 'x20' o 'x200'.")

    main_transform = standard_transform
    main_aug_transform = aug_transform
    return main_data_dir, secondary_data_dir, main_transform, main_aug_transform


def load_datasets():
    """
    1) Carga el dataset principal con transform estándar
    2) Divide en train/val (80/20)
    3) Crea train_aug solo con índices de train
    4) Concatena train + train_aug

    Retorna:
      full_dataset,
      train_dataset,
      val_dataset,
      train_aug_dataset,
      combined_train_dataset
    """
    main_data_dir, _secondary_data_dir, main_transform, main_aug_transform = get_dataset_configs()

    # Dataset original
    full_dataset = datasets.ImageFolder(main_data_dir, transform=main_transform)

    # Split reproducible
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Índices de train
    train_idx = train_dataset.indices

    # Dataset augmentado solo para train
    full_aug_dataset = datasets.ImageFolder(main_data_dir, transform=main_aug_transform)
    train_aug_dataset = Subset(full_aug_dataset, train_idx)

    # Concatena train limpio + train aumentado
    combined_train_dataset = ConcatDataset([train_dataset, train_aug_dataset])

    return full_dataset, train_dataset, val_dataset, train_aug_dataset, combined_train_dataset


def custom_collate(batch):
    """Stack images and labels into tensors for training"""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def get_dataloaders():
    """
    Configura y retorna los DataLoaders para entrenamiento y validación.

    Retorna:
      train_loader, val_loader
    """
    _, train_dataset, val_dataset, _train_aug, combined_train = load_datasets()

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
