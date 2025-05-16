from src.explicable_data_loader import get_dataloaders, load_datasets
from src.explicable_visualization import visualize_examples, visualize_pixel_distribution, print_class_distribution
from src.explicable_train import MultiModalResNet, hyperparameter_tuning
from config import DEVICE# Asegúrate de tener esto definido en config.py

def main():
    # Cargar datasets y DataLoaders
    full_dataset, train_dataset, val_dataset, train_aug_dataset, combined_train_dataset = load_datasets()
    train_loader, val_loader = get_dataloaders()

    class_to_idx = full_dataset.img_folder.class_to_idx
    print("Mapping de clases principal:", class_to_idx)
    print_class_distribution(full_dataset)

    # Visualizar ejemplos y distribución de píxeles
    visualize_examples(train_loader)
    dataiter = iter(train_loader)
    images, _, _ = next(dataiter)
    visualize_pixel_distribution(images)

    print("Tamaño del conjunto de entrenamiento combinado:", len(combined_train_dataset))
    print("Conjuntos incluidos en el conjunto de entrenamiento combinado:")
    for i, subset in enumerate(combined_train_dataset.datasets):
        print(f"\tTamaño del subconjunto {i + 1}:", len(subset))

    # Calcular el tamaño del conjunto de entrenamiento por clase
    class_counts = {class_name: 0 for class_name in class_to_idx.keys()}
    for _, _, label in combined_train_dataset:
        class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label)]
        class_counts[class_name] += 1

    print("Tamaño del conjunto de entrenamiento por clase:")
    for class_name, count in class_counts.items():
        print(f"Clase {class_name}: {count}")

    # Entrenar el modelo con búsqueda de hiperparámetros
    best_hparams = hyperparameter_tuning(train_dataset, val_dataset, full_dataset)

    print("Hiperparámetros óptimos:", best_hparams)

if __name__ == '__main__':
    main()
