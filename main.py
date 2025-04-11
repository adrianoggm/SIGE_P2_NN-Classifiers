import os
from src import utils  # Asumiendo que la carpeta 'src' está en el PYTHONPATH o en el mismo nivel

def main():
    # Obtener los datasets y dataloaders configurados
    (main_dataset, secondary_dataset, combined_train_dataset, 
     train_dataset_main, val_dataset_main, train_loader, val_loader) = utils.get_dataloaders(batch_size=32)
    
    # Mostrar el mapping de clases del dataset principal
    print("Mapping de clases principal:", main_dataset.class_to_idx)
    
    # Cargar y mostrar los archivos de descripción
    utils.load_descriptions()
    
    # Imprimir la distribución de imágenes en el dataset principal
    utils.print_class_distribution(main_dataset)
    
    # Visualizar una cuadrícula de ejemplos
    utils.visualize_examples(train_loader)
    
    # Tomar un batch y visualizar la distribución de píxeles
    dataiter = iter(train_loader)
    images, _ = next(dataiter)
    utils.visualize_pixel_distribution(images)
    
    # Aquí puedes continuar con el entrenamiento y validación usando train_loader y val_loader.
    # Por ejemplo, puedes iterar sobre las épocas e invocar el entrenamiento del modelo.

if __name__ == "__main__":
    main()
