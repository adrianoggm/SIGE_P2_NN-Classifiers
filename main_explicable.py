from src.explicable_data_loader import get_dataloaders, load_datasets
from src.explicable_visualization import visualize_examples, visualize_pixel_distribution, print_class_distribution
from src.explicable_train import MultiModalResNet, hyperparameter_tuning,train_model
from config import DEVICE# Asegúrate de tener esto definido en config.py

def main():
    # Leer variable de entorno para decidir si hacer fine-tuning
    import os
    fine_tune = os.getenv('FINE_TUNE', 'false').lower() in ('1', 'true', 'yes')

    # Cargar datasets y DataLoaders
    full_dataset, train_dataset, val_dataset, train_aug_dataset, combined_train_dataset = load_datasets()
    train_loader, val_loader = get_dataloaders()

    # Mostrar mapping y distribución
    class_to_idx = full_dataset.img_folder.class_to_idx
    print("Mapping de clases principal:", class_to_idx)
    print_class_distribution(full_dataset)

    # Visualización de ejemplos y distribución de píxeles
    visualize_examples(train_loader)
    images, attrs, labels = next(iter(train_loader))
    visualize_pixel_distribution(images)

    # Instanciar modelo multimodal
    try:
        num_classes = len(full_dataset.img_folder.class_to_idx)
    except AttributeError:
        num_classes = len(full_dataset.class_to_idx)
    attr_dim = list(full_dataset.image_attrs.values())[0].shape[0]
    model = MultiModalResNet(num_classes, attr_dim).to(DEVICE)

    if fine_tune:
        # Sólo fine-tuning de hiperparámetros (Optuna)
        print("Ejecutando fine-tuning de hiperparámetros con Optuna...")
        best_params = hyperparameter_tuning(
            train_dataset,
            val_dataset,
            full_dataset,
            n_trials=30
        )
        print("Mejores hiperparámetros encontrados:", best_params)
    else:
        # Entrenamiento estándar sin fine-tuning
        print("Entrenamiento estándar sin fine-tuning...")
        lr = 0.0006707369630642823
        optimizer = 'adam'
        final_val_acc = train_model(
            model,
            train_loader,
            val_loader,
            learning_rate=lr,
            optimizer_name=optimizer,
            save_best=True,
            use_wandb=True,
            with_htuning=False
        )
        print(f"Entrenamiento completo. Precisión final en validación: {final_val_acc:.2f}%")

if __name__ == '__main__':
    main()
