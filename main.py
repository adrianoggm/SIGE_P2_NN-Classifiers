from src.data_loader import get_dataloaders
from src.visualization import visualize_examples, visualize_pixel_distribution, print_class_distribution
from src.train import get_model, train_model
from config import DEVICE

def main():
    # Obtener datasets y DataLoaders
    main_dataset, secondary_dataset, combined_train_dataset, train_dataset_main, val_dataset_main, train_loader, val_loader = get_dataloaders()
    
    # Mostrar mapping de clases y distribución de datos
    print("Mapping de clases principal:", main_dataset.class_to_idx)
    print_class_distribution(main_dataset)
    
    # Visualizar ejemplos y distribución de píxeles
    visualize_examples(train_loader)
    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    visualize_pixel_distribution(images)
    
    # Configurar y entrenar el modelo
    num_classes = len(main_dataset.class_to_idx)
    model = get_model(num_classes)
    train_model(model, train_loader, val_loader)
    
if __name__ == '__main__':
    main()
