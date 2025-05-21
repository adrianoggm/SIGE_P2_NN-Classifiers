import os
import torch
BASE_DIR = os.path.dirname(__file__)
# Rutas para los datasets
DATA_DIR_X20 = './starting-package/data x20'
DATA_DIR_X200 = './starting-package/data x200'

ATTRIBUTES_PATH         = os.path.join(BASE_DIR, 'starting-package', 'data additional', 'attributes.txt')
IMAGE_ATTR_LABELS_PATH  = os.path.join(BASE_DIR, 'starting-package', 'data additional', 'image_attribute_labels.txt')
IMAGES_TXT_PATH         = os.path.join(BASE_DIR, 'starting-package', 'data additional', 'images.txt')

# Parámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Variable de entorno para elegir el dataset principal (por defecto "x20")
MAIN_DATASET = os.environ.get("MAIN_DATASET", "x200").lower()

# Configuración del dispositivo
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch_directml
DEVICE = torch_directml.device()    # esto apuntará a tu GPU AMD vía DirectML
