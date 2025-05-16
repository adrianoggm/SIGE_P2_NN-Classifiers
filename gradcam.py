#!/usr/bin/env python3
"""
Script para visualizar Grad-CAM sobre ejemplos del dataset CUB multimodal.
"""
import os
import torch
import matplotlib.pyplot as plt
from config import DEVICE
from src.explicable_data_loader import load_datasets, get_dataloaders
from src.explicable_train import MultiModalResNet, generate_gradcam


def main():
    # 1) Cargar datasets y dataloader de validación
    full_ds, train_ds, val_ds, train_aug_ds, combined_train_ds = load_datasets()
    _, val_loader = get_dataloaders()

    # 2) Configurar modelo y cargar pesos entrenados
    # Usar img_folder.class_to_idx para número de clases
    num_classes = len(full_ds.img_folder.class_to_idx)
    attr_dim = full_ds.attr_dim
    model = MultiModalResNet(num_classes, attr_dim).to(DEVICE)
    checkpoint_path = os.path.join(os.getcwd(), 'best_model_multimodal.pth')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # 3) Tomar un batch de validación
    dataiter = iter(val_loader)
    images, attrs, labels = next(dataiter)

    # 4) Visualizar Grad-CAM para los primeros ejemplos
    n_examples = min(6, images.size(0))
    plt.figure(figsize=(12, 8))
    for i in range(n_examples):
        img = images[i]
        attr = attrs[i]
        true_label = labels[i].item()

        # Generar Grad-CAM
        cam_img, pred_class = generate_gradcam(model, img, attr)

        # Plot original
        plt.subplot(2, n_examples, i+1)
        img_np = img.cpu().permute(1, 2, 0).numpy()
        plt.imshow(img_np.clip(0, 1))
        plt.title(f"True: {true_label}")
        plt.axis('off')

        # Plot GradCAM overlay
        plt.subplot(2, n_examples, n_examples + i+1)
        plt.imshow(cam_img)
        plt.title(f"GradCAM: {pred_class}")
        plt.axis('off')

    plt.suptitle("Grad-CAM en ejemplos de validación")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
