import matplotlib.pyplot as plt
import numpy as np

def visualize_examples(train_loader, num_examples=8):
    """
    Muestra una cuadrícula de ejemplos extraídos del train_loader para inspeccionar las transformaciones.
    """
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    plt.figure(figsize=(12, 6))
    cols = num_examples // 2  # Se asume una cuadrícula de 2 filas
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
