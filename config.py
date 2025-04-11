import os
import torch

# Rutas para los datasets
DATA_DIR_X20 = './starting-package/data x20'
DATA_DIR_X200 = './starting-package/data x200'

# Parámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Variable de entorno para elegir el dataset principal (por defecto "x20")
MAIN_DATASET = os.environ.get("MAIN_DATASET", "x20").lower()

# Configuración del dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
