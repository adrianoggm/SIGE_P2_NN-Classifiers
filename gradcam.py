#!/usr/bin/env python3
"""
Script para visualizar Grad-CAM y LIME sobre ejemplos del dataset CUB multimodal.
"""
import os
import torch
import matplotlib.pyplot as plt
from config import DEVICE
from src.explicable_data_loader import load_datasets, get_dataloaders
from src.explicable_train import MultiModalResNet, generate_gradcam, generate_lime_explanation, visualize_combined_explanations


def main():
    # 1) Cargar datasets y dataloader de validación
    full_ds, train_ds, val_ds, train_aug_ds, combined_train_ds = load_datasets()
    _, val_loader = get_dataloaders()

    # 2) Configurar modelo y cargar pesos entrenados
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

    # 4) Visualizar Grad-CAM y LIME para los primeros ejemplos
    n_examples = min(4, images.size(0))
    for i in range(n_examples):
        img = images[i]
        attr = attrs[i]
        true_label = labels[i].item()

        # Grad-CAM
        cam_img, pred_class = generate_gradcam(model, img, attr)

        # LIME
        lime_img, _ = generate_lime_explanation(model, img, attr, pred_class)

        # Visualización conjunta
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        img_np = img.cpu().permute(1, 2, 0).numpy()
        axs[0].imshow(img_np.clip(0, 1))
        axs[0].set_title(f"Original (True: {true_label})")
        axs[0].axis('off')

        axs[1].imshow(cam_img)
        axs[1].set_title(f"Grad-CAM (Pred: {pred_class})")
        axs[1].axis('off')

        axs[2].imshow(lime_img)
        axs[2].set_title(f"LIME (Pred: {pred_class})")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()