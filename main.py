from src.data_loader import get_dataloaders,load_datasets
from src.visualization import visualize_examples, visualize_pixel_distribution, print_class_distribution
from src.train import get_model, hyperparameter_tuning, hyperparameter_tuning_optuna, train_model
from config import DEVICE

def main():
    # Cargar datasets completos y DataLoaders
    full_dataset, train_dataset, val_dataset, train_aug_dataset, combined_train_dataset = load_datasets()
    train_loader, val_loader = get_dataloaders()

    # Mostrar mapping de clases y distribución global
    print("Mapping de clases principal:", full_dataset.class_to_idx)
    print_class_distribution(full_dataset)

    # Visualizar ejemplos y distribución de píxeles del conjunto de entrenamiento
    visualize_examples(train_loader)
    dataiter = iter(train_loader)
    images, _ = next(dataiter)

    # Mostrar la distribución de valores de píxeles de un lote de imágenes
    visualize_pixel_distribution(images)
    
    print("Tamaño del conjunto de entrenamiento combinado:", len(combined_train_dataset))
    print("Conjuntos incluidos en el conjunto de entrenamiento combinado:")
    for i, subset in enumerate(combined_train_dataset.datasets):
        print(f"\tTamaño del subconjunto {i + 1}:", len(subset))

    # Calcular y mostrar el tamaño del conjunto de entrenamiento por clase
    class_counts = {class_name: 0 for class_name in full_dataset.class_to_idx.keys()}
    for _, label in combined_train_dataset:
        class_name = list(full_dataset.class_to_idx.keys())[list(full_dataset.class_to_idx.values()).index(label)]
        class_counts[class_name] += 1

    print("Tamaño del conjunto de entrenamiento por clase:")
    for class_name, count in class_counts.items():
        print(f"Clase {class_name}: {count}")

    # Configurar y entrenar el modelo
    num_classes = len(full_dataset.class_to_idx)
    model = get_model(num_classes, model_type='resnet')


    
    train_model(model, train_loader, val_loader, learning_rate=1e-3, optimizer_name='adam', save_best=True, use_wandb=True, with_htuning=False)

    
    # best_hparams = hyperparameter_tuning(train_dataset,
    #                                     val_dataset,
    #                                     full_dataset,
    #                                     model_type='resnet',)
    # print("Hiperparámetros óptimos:", best_hparams)

    # best_config = hyperparameter_tuning_optuna(
    #     train_dataset, 
    #     val_dataset, 
    #     full_dataset,
    #     model_type='resnet',
    #     n_trials=30  # Número de configuraciones a probar
    # )

    # print("Hiperparámetros óptimos:", best_config)


if __name__ == '__main__':
    main()
