#!/usr/bin/env python3
"""
Script para calcular las medias y desviaciones estándar por canal de un dataset de imágenes.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from config import DATA_DIR_X20, DATA_DIR_X200, MAIN_DATASET


def get_main_data_dir():
    """
    Devuelve el directorio principal de imágenes según MAIN_DATASET.
    """
    if MAIN_DATASET == 'x20':
        return DATA_DIR_X20
    elif MAIN_DATASET == 'x200':
        return DATA_DIR_X200
    else:
        raise ValueError("MAIN_DATASET debe ser 'x20' o 'x200'.")


def compute_mean_std(dataset, batch_size=64, num_workers=4):
    """
    Recorre el dataset para calcular medias y desviaciones por canal.
    Retorna dos tensores (mean, std) de tamaño 3.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        # images: (B, 3, H, W)
        b, c, h, w = images.shape
        images = images.view(b, c, -1)  # (B, 3, H*W)
        sum_ += images.mean(dim=2).sum(dim=0)
        sum_sq += (images ** 2).mean(dim=2).sum(dim=0)
        total_images += b

    mean = sum_ / total_images
    var = sum_sq / total_images - mean ** 2
    std = torch.sqrt(var)
    return mean, std


def main():
    # 1) Directorio de imágenes
    data_dir = get_main_data_dir()
    print(f"Calculando estadísticas en: {data_dir}")

    # 2) Definir transform básico: Resize + ToTensor (0-1)
    basic_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # 3) Cargar dataset
    dataset = datasets.ImageFolder(data_dir, transform=basic_tf)

    # 4) Computar mean y std
    mean, std = compute_mean_std(dataset)

    # 5) Mostrar resultados
    mean_list = [float(m) for m in mean]
    std_list = [float(s) for s in std]
    print(f"Mean por canal: {mean_list}")
    print(f"Std  por canal: {std_list}")


if __name__ == '__main__':
    main()
