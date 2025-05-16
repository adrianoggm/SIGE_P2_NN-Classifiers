import matplotlib.pyplot as plt
import numpy as np

def visualize_examples(train_loader, num_examples=12):
    """
    Muestra una cuadrícula de ejemplos extraídos del train_loader para inspeccionar las transformaciones.
    """
    dataiter = iter(train_loader)
    # Desempaquetar imágenes, atributos (ignorados) y etiquetas
    images, attrs, labels = next(dataiter)

    plt.figure(figsize=(12, 6))
    cols = num_examples // 2
    for idx in range(min(num_examples, len(images))):
        plt.subplot(2, cols, idx + 1)
        img = images[idx].cpu().permute(1, 2, 0)  # De (C, H, W) a (H, W, C)
        plt.imshow(img.clip(0, 1))
        plt.title(f"Label: {labels[idx]}")
        plt.axis('off')
    plt.suptitle("Ejemplos del DataLoader (tamaños variables)")
    plt.show()


def visualize_pixel_distribution(images):
    """
    Muestra un histograma de la distribución de valores de píxeles a partir de un batch de imágenes.
    """
    all_pixels = []
    for img in images:
        all_pixels.extend(img.cpu().numpy().flatten())
    all_pixels = np.array(all_pixels)

    plt.figure(figsize=(8, 4))
    plt.hist(all_pixels, bins=50)
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
    for label, count in sorted(class_counts.items()):
        print(f"Clase {label}: {count} imágenes")

    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Clase")
    plt.ylabel("Número de imágenes")
    plt.title("Distribución de imágenes por clase")
    plt.xticks(rotation=45)
    plt.show()
