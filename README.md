# SIGE_P2_NN-Classifiers: Terminal Guide

## 📚 Overview

This repository was developed for the **SIGE** course. Its goal is to build a **robust bird image classifier**, handling two different resolutions: **20x20** and **200x200**.

The project is modularized into the following components:

- `config.py` — Global configurations (paths, training parameters, environment settings).
- `src/data_loader.py` — Data loading and preprocessing (transformations, dataset splitting, DataLoaders).
- `src/visualization.py` — Visualization tools (examples, pixel distributions, class counts).
- `src/train.py` — Model definition, training loop, and validation (fine-tuning a pre-trained model).
- `main.py` — Orchestrates the entire workflow (data loading, visualization, training).

---

## ⚙️ Installation

1. Create a virtual environment:


```bash
python -m venv venv
```
2. Activate the virtual environment:


**On Windows:**
```bash
.\venv\Scripts\activate
``` 
**On macOS/Linux:**
```bash
source venv/bin/activate
```
Install the required dependencies
```bash
pip install -r requirements.txt
```
⚠️  Make sure requirements.txt includes packages like:
    - torch
    - torchvision
    - matplotlib
    - etc.

🛠️  Configuration
-----------------
File: config.py

  Dataset directories:
    • DATA_DIR_X20   → Path to 20x20 image dataset
    • DATA_DIR_X200  → Path to 200x200 image dataset

  Training parameters:
    • BATCH_SIZE
    • EPOCHS
    • LEARNING_RATE

  MAIN_DATASET environment variable:
    • If 'x20'  → 200x200 images are downscaled
    • If 'x200' → 20x20 images are upscaled


🧪 Data Loading & Preprocessing
-------------------------------
Module: src/data_loader.py

  Transformation pipelines (torchvision.transforms):
    • Standard Transform     → Resize to 224x224
    • Augmentation Transform → Random crops, flips, rotations
    • Scaling Transforms:
        - Downscaling → 112x112
        - Upscaling   → 448x448

  Dataset loading:
    • Using ImageFolder
    • Combined with ConcatDataset for augmented versions

  Dataset split:
    • 80% Training
    • 20% Validation


📊 Visualization
----------------
Module: src/visualization.py

  • Display sample image grid (verify augmentations and scaling)
  • Plot pixel value distribution histograms
  • Print class distribution in dataset


🧠 Training
-----------
Module: src/train.py

  • Fine-tune pre-trained ResNet18 for bird classification
  • Training loop outputs:
      - Training loss
      - Validation accuracy
    (for each epoch)

(venv) PS C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers> python .\main_explicable.py
Mapping de clases principal: {'001.Black_footed_Albatross': 0, '002.Laysan_Albatross': 1, '003.Sooty_Albatross': 2, '004.Groove_billed_Ani': 3, '005.Crested_Auklet': 4, '006.Least_Auklet': 5, '007.Parakeet_Auklet': 6, '008.Rhinoceros_Auklet': 7, '009.Brewer_Blackbird': 8, '010.Red_winged_Blackbird': 9, '011.Rusty_Blackbird': 10, '012.Yellow_headed_Blackbird': 11, '013.Bobolink': 12, '014.Indigo_Bunting': 13, '015.Lazuli_Bunting': 14, '016.Painted_Bunting': 15, '017.Cardinal': 16, '018.Spotted_Catbird': 17, '019.Gray_Catbird': 18, '020.Yellow_breasted_Chat': 19}
Distribución de imágenes en el dataset principal:
Clase 0: 60 imágenes
Clase 1: 60 imágenes
Clase 2: 58 imágenes
Clase 3: 60 imágenes
Clase 4: 44 imágenes
Clase 5: 41 imágenes
Clase 6: 53 imágenes
Clase 7: 48 imágenes
Clase 8: 59 imágenes
Clase 9: 60 imágenes
Clase 10: 60 imágenes
Clase 11: 56 imágenes
Clase 12: 60 imágenes
Clase 13: 60 imágenes
Clase 14: 58 imágenes
Clase 15: 58 imágenes
Clase 16: 57 imágenes
Clase 17: 45 imágenes
Clase 18: 59 imágenes
Clase 19: 59 imágenes
Tamaño del conjunto de entrenamiento combinado: 1784
Conjuntos incluidos en el conjunto de entrenamiento combinado:
        Tamaño del subconjunto 1: 892
        Tamaño del subconjunto 2: 892
Tamaño del conjunto de entrenamiento por clase:
Clase 001.Black_footed_Albatross: 98
Clase 002.Laysan_Albatross: 92
Clase 003.Sooty_Albatross: 98
Clase 004.Groove_billed_Ani: 92
Clase 005.Crested_Auklet: 66
Clase 006.Least_Auklet: 64
Clase 007.Parakeet_Auklet: 78
Clase 008.Rhinoceros_Auklet: 74
Clase 009.Brewer_Blackbird: 92
Clase 010.Red_winged_Blackbird: 96
Clase 011.Rusty_Blackbird: 98
Clase 012.Yellow_headed_Blackbird: 98
Clase 013.Bobolink: 88
Clase 014.Indigo_Bunting: 100
Clase 015.Lazuli_Bunting: 94
Clase 016.Painted_Bunting: 96
Clase 017.Cardinal: 94
Clase 018.Spotted_Catbird: 74
Clase 019.Gray_Catbird: 96
Clase 020.Yellow_breasted_Chat: 96
Traceback (most recent call last):
  File "C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\main_explicable.py", line 41, in <module>
    main()
  File "C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\main_explicable.py", line 37, in main
    best_hparams = hyperparameter_tuning(train_dataset, val_dataset, full_dataset, attr_dim=full_dataset.attr_dim)
TypeError: hyperparameter_tuning() got an unexpected keyword argument 'attr_dim'
(venv) PS C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers> python .\main_explicable.py
Mapping de clases principal: {'001.Black_footed_Albatross': 0, '002.Laysan_Albatross': 1, '003.Sooty_Albatross': 2, '004.Groove_billed_Ani': 3, '005.Crested_Auklet': 4, '006.Least_Auklet': 5, '007.Parakeet_Auklet': 6, '008.Rhinoceros_Auklet': 7, '009.Brewer_Blackbird': 8, '010.Red_winged_Blackbird': 9, '011.Rusty_Blackbird': 10, '012.Yellow_headed_Blackbird': 11, '013.Bobolink': 12, '014.Indigo_Bunting': 13, '015.Lazuli_Bunting': 14, '016.Painted_Bunting': 15, '017.Cardinal': 16, '018.Spotted_Catbird': 17, '019.Gray_Catbird': 18, '020.Yellow_breasted_Chat': 19}
Distribución de imágenes en el dataset principal:
Clase 0: 60 imágenes
Clase 1: 60 imágenes
Clase 2: 58 imágenes
Clase 3: 60 imágenes
Clase 4: 44 imágenes
Clase 5: 41 imágenes
Clase 6: 53 imágenes
Clase 7: 48 imágenes
Clase 8: 59 imágenes
Clase 9: 60 imágenes
Clase 10: 60 imágenes
Clase 11: 56 imágenes
Clase 12: 60 imágenes
Clase 13: 60 imágenes
Clase 14: 58 imágenes
Clase 15: 58 imágenes
Clase 16: 57 imágenes
Clase 17: 45 imágenes
Clase 18: 59 imágenes
Clase 19: 59 imágenes
Tamaño del conjunto de entrenamiento combinado: 1784
Conjuntos incluidos en el conjunto de entrenamiento combinado:
        Tamaño del subconjunto 1: 892
        Tamaño del subconjunto 2: 892
Tamaño del conjunto de entrenamiento por clase:
Clase 001.Black_footed_Albatross: 98
Clase 002.Laysan_Albatross: 92
Clase 003.Sooty_Albatross: 98
Clase 004.Groove_billed_Ani: 92
Clase 005.Crested_Auklet: 66
Clase 006.Least_Auklet: 64
Clase 007.Parakeet_Auklet: 78
Clase 008.Rhinoceros_Auklet: 74
Clase 009.Brewer_Blackbird: 92
Clase 010.Red_winged_Blackbird: 96
Clase 011.Rusty_Blackbird: 98
Clase 012.Yellow_headed_Blackbird: 98
Clase 013.Bobolink: 88
Clase 014.Indigo_Bunting: 100
Clase 015.Lazuli_Bunting: 94
Clase 016.Painted_Bunting: 96
Clase 017.Cardinal: 94
Clase 018.Spotted_Catbird: 74
Clase 019.Gray_Catbird: 96
Clase 020.Yellow_breasted_Chat: 96
Traceback (most recent call last):
  File "C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\main_explicable.py", line 42, in <module>
    main()
  File "C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\main_explicable.py", line 37, in main
    best_hparams = hyperparameter_tuning(train_dataset, val_dataset, full_dataset)
  File "C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\src\explicable_train.py", line 110, in hyperparameter_tuning
    num_classes = len(full_dataset.class_to_idx)
AttributeError: 'CUBMultimodalDataset' object has no attribute 'class_to_idx'
(venv) PS C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers> python .\main_explicable.py
Mapping de clases principal: {'001.Black_footed_Albatross': 0, '002.Laysan_Albatross': 1, '003.Sooty_Albatross': 2, '004.Groove_billed_Ani': 3, '005.Crested_Auklet': 4, '006.Least_Auklet': 5, '007.Parakeet_Auklet': 6, '008.Rhinoceros_Auklet': 7, '009.Brewer_Blackbird': 8, '010.Red_winged_Blackbird': 9, '011.Rusty_Blackbird': 10, '012.Yellow_headed_Blackbird': 11, '013.Bobolink': 12, '014.Indigo_Bunting': 13, '015.Lazuli_Bunting': 14, '016.Painted_Bunting': 15, '017.Cardinal': 16, '018.Spotted_Catbird': 17, '019.Gray_Catbird': 18, '020.Yellow_breasted_Chat': 19}
Distribución de imágenes en el dataset principal:
Clase 0: 60 imágenes
Clase 1: 60 imágenes
Clase 2: 58 imágenes
Clase 3: 60 imágenes
Clase 4: 44 imágenes
Clase 5: 41 imágenes
Clase 6: 53 imágenes
Clase 7: 48 imágenes
Clase 8: 59 imágenes
Clase 9: 60 imágenes
Clase 10: 60 imágenes
Clase 11: 56 imágenes
Clase 12: 60 imágenes
Clase 13: 60 imágenes
Clase 14: 58 imágenes
Clase 15: 58 imágenes
Clase 16: 57 imágenes
Clase 17: 45 imágenes
Clase 18: 59 imágenes
Clase 19: 59 imágenes
Tamaño del conjunto de entrenamiento combinado: 1784
Conjuntos incluidos en el conjunto de entrenamiento combinado:
        Tamaño del subconjunto 1: 892
        Tamaño del subconjunto 2: 892
Tamaño del conjunto de entrenamiento por clase:
Clase 001.Black_footed_Albatross: 98
Clase 002.Laysan_Albatross: 92
Clase 003.Sooty_Albatross: 98
Clase 004.Groove_billed_Ani: 92
Clase 005.Crested_Auklet: 66
Clase 006.Least_Auklet: 64
Clase 007.Parakeet_Auklet: 78
Clase 008.Rhinoceros_Auklet: 74
Clase 009.Brewer_Blackbird: 92
Clase 010.Red_winged_Blackbird: 96
Clase 011.Rusty_Blackbird: 98
Clase 012.Yellow_headed_Blackbird: 98
Clase 012.Yellow_headed_Blackbird: 98
Clase 013.Bobolink: 88
Clase 014.Indigo_Bunting: 100
Clase 015.Lazuli_Bunting: 94
Clase 015.Lazuli_Bunting: 94
Clase 016.Painted_Bunting: 96
Clase 017.Cardinal: 94
Clase 018.Spotted_Catbird: 74
Clase 019.Gray_Catbird: 96
Clase 020.Yellow_breasted_Chat: 96
Probando: lr=0.001, batch_size=32, optimizer=adam
C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\venv\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\adria\OneDrive\Documentos\SIGE_P2_NN-Classifiers\venv\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.      
  warnings.warn(msg)
Epoch 1/10 - Val Acc: 22.42%
Epoch 2/10 - Val Acc: 36.32%
Epoch 3/10 - Val Acc: 29.60%
Epoch 4/10 - Val Acc: 48.43%
Epoch 5/10 - Val Acc: 28.25%
Epoch 6/10 - Val Acc: 68.61%
Epoch 7/10 - Val Acc: 71.30%
Epoch 8/10 - Val Acc: 61.43%
Epoch 9/10 - Val Acc: 60.99%
Epoch 10/10 - Val Acc: 65.02%
Validación: 71.30%
Probando: lr=0.001, batch_size=32, optimizer=sgd
Epoch 1/10 - Val Acc: 16.14%
Epoch 2/10 - Val Acc: 34.98%
Epoch 3/10 - Val Acc: 54.26%
Epoch 4/10 - Val Acc: 62.33%
Epoch 5/10 - Val Acc: 72.65%
Epoch 6/10 - Val Acc: 78.03%
Epoch 7/10 - Val Acc: 82.06%
Epoch 8/10 - Val Acc: 82.06%
Epoch 9/10 - Val Acc: 81.61%
Epoch 10/10 - Val Acc: 82.06%
Validación: 82.06%
Probando: lr=0.001, batch_size=64, optimizer=adam
Epoch 1/10 - Val Acc: 43.95%
Epoch 2/10 - Val Acc: 66.82%
Epoch 3/10 - Val Acc: 47.53%
Epoch 4/10 - Val Acc: 64.57%
Epoch 5/10 - Val Acc: 62.78%
Epoch 6/10 - Val Acc: 62.78%
Epoch 7/10 - Val Acc: 76.68%
Epoch 8/10 - Val Acc: 69.51%
Epoch 9/10 - Val Acc: 52.02%
Epoch 10/10 - Val Acc: 48.43%
Validación: 76.68%
Probando: lr=0.001, batch_size=64, optimizer=sgd
Epoch 1/10 - Val Acc: 8.97%
Epoch 2/10 - Val Acc: 21.52%
Epoch 3/10 - Val Acc: 34.08%
Epoch 4/10 - Val Acc: 47.53%
Epoch 5/10 - Val Acc: 49.33%
Epoch 6/10 - Val Acc: 55.16%
Epoch 7/10 - Val Acc: 60.54%
Epoch 8/10 - Val Acc: 65.02%
Epoch 9/10 - Val Acc: 67.26%
Epoch 10/10 - Val Acc: 71.75%
Validación: 71.75%
Probando: lr=0.0001, batch_size=32, optimizer=adam
Epoch 1/10 - Val Acc: 61.88%
Epoch 2/10 - Val Acc: 75.34%
Epoch 3/10 - Val Acc: 81.17%
Epoch 4/10 - Val Acc: 83.41%
Epoch 5/10 - Val Acc: 82.51%
Epoch 6/10 - Val Acc: 85.20%
Epoch 7/10 - Val Acc: 84.75%
Epoch 8/10 - Val Acc: 84.30%
Epoch 9/10 - Val Acc: 83.41%
Epoch 10/10 - Val Acc: 85.20%
Validación: 85.20%
Probando: lr=0.0001, batch_size=32, optimizer=sgd
Epoch 1/10 - Val Acc: 8.07%
Epoch 2/10 - Val Acc: 10.31%
Epoch 3/10 - Val Acc: 12.56%
Epoch 4/10 - Val Acc: 14.35%
Epoch 5/10 - Val Acc: 17.49%
Epoch 6/10 - Val Acc: 18.39%
Epoch 7/10 - Val Acc: 22.87%
Epoch 8/10 - Val Acc: 25.11%
Epoch 9/10 - Val Acc: 26.91%
Epoch 10/10 - Val Acc: 28.25%
Validación: 28.25%
Probando: lr=0.0001, batch_size=64, optimizer=adam
Epoch 1/10 - Val Acc: 41.26%
Epoch 2/10 - Val Acc: 69.51%
Epoch 3/10 - Val Acc: 79.37%
Epoch 4/10 - Val Acc: 79.82%
Epoch 5/10 - Val Acc: 82.96%
Epoch 6/10 - Val Acc: 84.30%
Epoch 7/10 - Val Acc: 86.10%
Epoch 8/10 - Val Acc: 85.20%
Epoch 9/10 - Val Acc: 86.10%
Epoch 10/10 - Val Acc: 86.55%
Validación: 86.55%
Probando: lr=0.0001, batch_size=64, optimizer=sgd
Epoch 1/10 - Val Acc: 4.48%
Epoch 2/10 - Val Acc: 5.38%
Epoch 3/10 - Val Acc: 6.28%
Epoch 4/10 - Val Acc: 6.73%
Epoch 5/10 - Val Acc: 7.17%
Epoch 6/10 - Val Acc: 8.52%
Epoch 7/10 - Val Acc: 9.87%
Epoch 8/10 - Val Acc: 10.76%
Epoch 9/10 - Val Acc: 11.21%
Epoch 10/10 - Val Acc: 12.56%
Validación: 12.56%
Mejor configuración: {'learning_rate': 0.0001, 'batch_size': 64, 'optimizer': 'adam'} con Val Acc: 86.55%
Epoch 1/10 - Val Acc: 38.57%
Epoch 2/10 - Val Acc: 67.26%
Epoch 3/10 - Val Acc: 78.48%
Epoch 4/10 - Val Acc: 78.03%
Epoch 5/10 - Val Acc: 82.96%
Epoch 6/10 - Val Acc: 83.41%
Epoch 7/10 - Val Acc: 82.51%
Epoch 8/10 - Val Acc: 82.96%
Epoch 9/10 - Val Acc: 83.86%
Epoch 10/10 - Val Acc: 84.75%
Hiperparámetros óptimos: {'learning_rate': 0.0001, 'batch_size': 64, 'optimizer': 'adam'}



Diseño de un Modelo de Clasificación Multimodal Explicable para CUB-200-2011
1. Introducción
El conjunto de datos CUB-200-2011 (Caltech-UCSD Birds-200-2011) es un benchmark de clasificación fine-grained de especies de aves. Contiene 200 categorías de pájaros con un total de 11.788 imágenes, junto con anotaciones detalladas como 312 atributos binarios por imagen (por ejemplo, colores de partes del ave)
paperswithcode.com
. Además, incluye ubicaciones de partes (picos, alas, etc.) y cuadros delimitadores, aunque en este caso nos centramos en los atributos. El problema que abordamos es diseñar un enfoque multimodal que combine la información visual de las imágenes con las características adicionales (atributos) proporcionadas en un archivo de texto (formato "etiqueta : nombre_atributo: valor" como '... : has_wing_color: blue' o '... : has_primary_shape: 23,24'). El objetivo es lograr una clasificación más precisa de la especie de ave y, a la vez, que el modelo pueda justificar sus decisiones de forma comprensible para humanos. La clasificación fine-grained de aves es un reto debido a las sutiles diferencias visuales entre especies (p. ej., variaciones de color o forma en las alas o el pico). Incluir atributos etiquetados manualmente (como color de alas = azul) puede mejorar el reconocimiento, actuando como pistas de alto nivel. Un enfoque que integre ambos tipos de datos promete no solo mayor exactitud sino también mejor interpretabilidad, pues las predicciones pueden explicarse en términos de características semánticas (ej. "alas azules") además de evidencias visuales. Investigaciones previas han demostrado que modelos que utilizan conceptos/atributos humanos alcanzan desempeños competitivos con redes neuronales end-to-end, al tiempo que permiten interpretaciones en términos de dichos atributos
github.com
. En este informe se propone una arquitectura multimodal que combina una CNN para imágenes con una red densa para los atributos, detallando cómo preprocesar y vectorizar estos atributos. También se discuten diversas técnicas de explicabilidad para entender las predicciones del modelo: desde mapas de calor visuales (Grad-CAM, LIME) hasta explicaciones basadas en atributos o reglas simbólicas. Por último, se mencionan herramientas prácticas para implementar la solución (PyTorch, TensorFlow, scikit-learn, etc.) y criterios para evaluar tanto el rendimiento predictivo como la calidad de las explicaciones generadas.
2. Arquitectura Multimodal Propuesta


Figura 1: Esquema de la arquitectura propuesta. La arquitectura integrada consta de dos ramas principales que procesan cada modalidad por separado, fusionándose antes de la predicción final. En la primera rama, una red neuronal convolucional (CNN) procesa la imagen de entrada y extrae un vector de features visuales de alto nivel. En paralelo, la segunda rama procesa los atributos textuales asociados (características estructuradas) convirtiéndolos en un vector numérico. Estas dos representaciones se concatenan para formar una representación conjunta, que luego alimenta a capas completamente conectadas (fully connected) que realizan la clasificación de la especie de ave. En términos concretos, la rama visual puede ser una CNN pre-entrenada (por ejemplo, ResNet, VGG o EfficientNet) truncada antes de la capa de clasificación, de modo que su salida sea un embebido visual de dimensión fija (p. ej. 2048 dimensiones en ResNet50). La rama de atributos puede consistir en una pequeña red feed-forward: por ejemplo, una o dos capas densas que toman como entrada el vector de atributos preprocesados (ver sección de Preprocesamiento) y producen un embebido de atributos de, digamos, 256 dimensiones. La fusión multimodal se realiza típicamente en la etapa intermedia o final: la opción más sencilla es concatenar el vector de características visuales y el vector de atributos, y luego pasar este vector combinado por una o más capas densas que emitan las 200 clases mediante softmax. Este enfoque de combinar CNN con datos tabulares es una práctica común y efectiva
stackoverflow.com
: se aprovecha la capacidad de la CNN para aprender representaciones visuales y del MLP (perceptrón multicapa) para los atributos, uniéndolos en un modelo único. Alternativas de fusión: Una variante posible es la fusión temprana, donde los atributos influyen en etapas intermedias de la CNN (por ejemplo, concatenando el vector de atributos con las activaciones de alguna capa oculta de la CNN en lugar de solo al final). Sin embargo, una fusión tardía (post-extracción de características) suele ser más sencilla de entrenar y aprovechar las modalidades por separado. Otra alternativa interesante es utilizar un enfoque de bottleneck de conceptos: primero entrenar un modelo para predecir los atributos a partir de la imagen, y luego otro modelo (o capa) que prediga la especie a partir de esos atributos predichos
github.com
. Este enfoque, conocido como Concept Bottleneck Model (CBM), hace que la predicción pase por la capa de atributos como representación intermedia interpretable. En la práctica se puede implementar de manera conjunta: la CNN primero produce predicciones de los 312 atributos (con supervisión de los atributos reales en entrenamiento), luego esas predicciones se usan como entrada a la capa de clasificación final. Esto garantiza que las decisiones se basen explícitamente en los atributos, facilitando explicaciones (ver sección Explicabilidad). No obstante, los CBM pueden sacrificar algo de precisión si los atributos predichos no son perfectos. Por ello, una opción híbrida es entrenar el modelo multimodal end-to-end con ambas pérdidas: pérdida de clasificación de especie y, opcionalmente, pérdida de reconstrucción de atributos, para alentar al modelo a prestar atención a esas características. En resumen, la arquitectura propuesta integra eficientemente las dos fuentes de datos: la imagen proporciona información visual detallada y los atributos aportan conocimiento estructurado adicional. Esta combinación multimodal debería mejorar la precisión al resolver ambigüedades visuales con pistas textuales, a la vez que establece la base para explicaciones más transparentes, ya que el modelo maneja conceptos entendibles (los atributos) internamente.
3. Preprocesamiento y Vectorización de Atributos
Los atributos suministrados en el dataset vienen en formato texto con la sintaxis "etiqueta: nombre_atributo: valor". Antes de alimentar estos datos al modelo, es necesario transformarlos en un vector numérico manejable. Suponiendo que cada imagen (o cada instancia) tiene una serie de líneas describiendo propiedades, el primer paso es parsear el archivo de atributos. Por ejemplo, consideremos las entradas:
12 : has_wing_color: blue
12 : has_primary_shape: 23,24
Estas líneas podrían indicar que para la imagen con etiqueta 12 (especie particular), el color de las alas es azul y tiene cierta forma primaria identificada por índices 23 y 24. En el CUB, los atributos suelen ser binarios predefinidos (312 atributos posibles, e.g. "tiene alas azules = sí/no")
paperswithcode.com
. En la práctica, podemos construir un vocabulario de atributos a partir de todos los pares nombre_atributo=valor presentes en el conjunto de datos. Cada combinación única se convertirá en una dimensión en el vector de atributos. En este caso, podríamos descomponer has_wing_color: blue en un atributo binario "wing_color_blue" que vale 1 para esta ave. De forma similar, si has_wing_color: red aparece en otra instancia, tendríamos "wing_color_red" como otra característica. Para atributos numéricos o con múltiples valores (como has_primary_shape: 23,24), interpretaremos que la propiedad primary_shape incluye las categorías 23 y 24. Esto se puede representar activando dos características binarias: por ejemplo "primary_shape_23" = 1 y "primary_shape_24" = 1 para esa imagen. En general, cada atributo puede procesarse así:
Atributos categóricos (texto): convertir a one-hot encoding. Cada posible valor de ese atributo se vuelve un componente binario. Ej: atributo "wing_color" con posibles valores {blue, brown, black,...} genera varias columnas (wing_color_blue, wing_color_brown, etc.), colocando 1 en la correspondiente al valor presente y 0 en las demás.
Atributos multi-valor: si un atributo puede tener lista de valores (como múltiples partes presentes), se activan todas las columnas correspondientes. (Alternativamente, se podría asignar múltiples valores a un embedding, pero dado que son pocos y fijos es más simple multi-hot).
Atributos numéricos continuos: si existieran (ej: longitud de ala = 5.2 cm), se pueden normalizar o discretizar según convenga. (En CUB, prácticamente todos los atributos son discretos/categóricos o booleanos).
Tras este proceso, cada imagen queda representada por un vector de atributos de dimensión fija (la cantidad total de atributos únicos en el dataset). Con CUB, ese tamaño será del orden de 312 (si usamos directamente los 312 atributos binarios originales) hasta algunos cientos más si diferenciamos valores distintos de un mismo atributo como features separadas. Dado que el dataset ya define 312 atributos binarios, una buena práctica es utilizar esa estructura: por ejemplo, los atributos podrían incluir cosas como wing color = blue, wing pattern = striped, bill shape = curved, etc., cada uno binario indicando presencia/ausencia de esa característica en la imagen. Podemos mapear el texto a esos índices conocidos. Si el archivo no da directamente el vector de 312 bits por imagen, tendremos que generarlo: inicializar un vector de 312 ceros y poner 1 en las posiciones correspondientes a los atributos mencionados en las líneas para esa imagen. Una vez vectorizados los atributos, conviene escalarlos o normalizarlos si son continuos; en caso de atributos binarios o one-hot, no se requiere escalado especial (0/1 ya está bien). También es útil dividir los datos en entrenamiento/validación/test asegurando que para cada imagen la fila de atributos se alinea con la imagen en el modelo (por ejemplo, usando el mismo ID o nombre de archivo para hacer join entre la tabla de atributos y las imágenes). Durante el entrenamiento, podremos alimentar al modelo pares (imagen, vector_atributos) como entrada conjunta, mientras que en inferencia se requeriría disponer de los atributos para cada nueva imagen. Si en una aplicación real no se tuvieran los atributos manualmente, podríamos primero predecirlos con un sub-modelo, como se comentó en el esquema de concept bottleneck. En resumen, el preprocesamiento transforma las descripciones como "has_wing_color: blue" en variables numéricas utilizables por la red. Esto permite que la información semántica de los atributos se integre con las características visuales aprendidas por la CNN, aportando contexto adicional para diferenciar especies parecidas (por ejemplo, dos aves visualmente similares podrían distinguirse porque una tiene patas rojas y otra patas amarillas, información que el vector de atributos explicitamente codifica).
4. Explicabilidad del Modelo
Una vez entrenado el modelo multimodal, es crucial disponer de mecanismos que expliquen por qué se tomó una decisión determinada. La explicabilidad se puede abordar desde dos frentes complementarios: (1) explicaciones visuales que señalan qué partes de la imagen influyeron en la clasificación, y (2) explicaciones simbólicas o basadas en atributos que describen la predicción en términos de las características de alto nivel (color, forma, etc.). A continuación, describimos varias técnicas en ambas categorías y otras adicionales, junto con recomendaciones para su uso.
4.1 Explicaciones Visuales (saliency maps)
Grad-CAM (Gradient-weighted Class Activation Mapping) – Es una técnica popular para obtener una visualización de las áreas de la imagen a las que la CNN prestó más atención para predecir una cierta clase. Grad-CAM calcula la importancia de cada región a partir de los gradientes de la clase objetivo con respecto a las activaciones de las últimas capas convolucionales. El resultado es un mapa de calor sobre la imagen original, donde las zonas más calientes (rojo/intenso) indican mayor contribución a la predicción de esa clase
adataodyssey.com
. En esencia, Grad-CAM destaca cuáles partes de la imagen fueron usadas por el modelo para clasificar la imagen como una especie particular. Por ejemplo, si el modelo identifica correctamente un Blue Jay, Grad-CAM podría resaltar la región de las alas o el pecho azul del ave, mostrando que esos píxeles influyeron fuertemente en la decisión. Este método es independiente del modelo en el sentido de que funciona con cualquier CNN con arquitectura con capas convolucionales finales (no requiere modificar el entrenamiento). Es rápido de calcular y provee una explicación visual intuitiva – para un humano es fácil verificar si el modelo miró al pájaro (y qué parte del pájaro) o si se distrajo con el fondo. LIME (Local Interpretable Model-Agnostic Explanations) – Mientras Grad-CAM es específico de CNNs, LIME es un método agnóstico al modelo que puede aplicarse también a imágenes. En el caso de imágenes, LIME genera explicaciones locales entrenando un modelo interpretable (por ejemplo, un modelo lineal con pocas variables) en la vecindad de la instancia a explicar
vishnudsharma.medium.com
proceedings.mlr.press
. ¿Cómo se logra esto? Primero, LIME segmenta la imagen en regiones significativas llamadas superpíxeles (grupos de píxeles contiguos con color/textura similar)
vishnudsharma.medium.com
. Cada superpíxel se trata como una característica interpretable que puede estar "presente" u "oculta". Luego LIME genera muchas versiones de la imagen perturbadas al azar, encendiendo y apagando superpíxeles (p. ej., quitando o griseando ciertas regiones)
proceedings.mlr.press
. Para cada versión perturbada obtiene la predicción del modelo original (la probabilidad de cada clase). Con ese conjunto de datos (presencia/ausencia de regiones -> resultado del modelo), LIME ajusta un modelo lineal aproximado que predice la output del modelo en función de qué superpíxeles están presentes. Las ponderaciones de ese modelo lineal sirven como explicación: indican cuáles segmentos de la imagen contribuyen positiva o negativamente a la clasificación actual. LIME normalmente resaltará, por ejemplo, 3-5 regiones de la imagen: marcando en verde las que más aportan a favor de la predicción y en rojo las que van en contra. En el contexto de CUB, LIME podría, por ejemplo, señalar que la región correspondiente a las alas y cola (de color azul) aportó positivamente a predecir Blue Jay, mientras que quizá la región del fondo cielo neutro no fue importante. LIME es útil porque no depende de acceder a gradientes internos, funciona con cualquier modelo como caja negra, y permite también explicar fallos del modelo al ver qué partes lo confundieron. Como desventaja, es computacionalmente más costoso (muchas evaluaciones del modelo con perturbaciones) y tiene cierta variabilidad en cada ejecución local. Otras técnicas visuales – Existen otros métodos de saliency o mapas de importancia a nivel de píxel similares a Grad-CAM y LIME. Por ejemplo, Integrated Gradients calcula la contribución de cada píxel integrando los gradientes mientras se varía la imagen desde un estado de referencia (como una imagen negra) hasta la actual. El resultado es otro mapa de relevancia de pixels, más fino que Grad-CAM (que opera a nivel de característica de alto nivel) y que satisface propiedades teóricas de aditividad. También se puede emplear Occlusion analysis, que consiste en ocluir (enmascarar) sistemáticamente diferentes partes de la imagen y medir cuánto baja la confianza en la predicción; las regiones cuya ausencia más reduce la puntuación de la clase son las más importantes. Métodos basados en attention (si la arquitectura tuviera mecanismos de atención) también permiten visualizar a qué pixeles o regiones atendió el modelo. En general, la recomendación es utilizar Grad-CAM como primera opción por su facilidad y rapidez, complementándolo con LIME o Integrated Gradients para validación. Por ejemplo, si Grad-CAM y LIME ambos señalan las alas azules, tenemos alta confianza de que esa característica visual es clave para la predicción.
4.2 Explicaciones basadas en Atributos y Reglas Simbólicas
Dado que nuestro modelo incorpora atributos explícitamente, podemos explotar esto para obtener explicaciones simbólicas más cercanas al razonamiento humano. La idea central es expresar la decisión del modelo en términos de los atributos de alto nivel (color, forma, etc.) que describen al ave. Una forma sencilla de hacerlo es inspeccionando la contribución de cada atributo en el modelo. Si la rama de atributos en la red termina en una capa totalmente conectada hacia las clases, los pesos de esa capa indican cuánto influencia tiene cada atributo en cada clase. Por ejemplo, supongamos que para la clase Blue Jay el peso asociado al atributo wing_color_blue es muy alto; eso sugiere que tener alas azules es un fuerte indicador de esa especie (lo cual intuitivamente es cierto). De modo similar, si el atributo has_primary_color_red tiene peso negativo para Blue Jay pero positivo para Northern Cardinal, refleja que el modelo entiende que el cardinal es rojo mientras el Blue Jay no. Estas relaciones peso-clase pueden traducirse en explicaciones del tipo: "El modelo predijo Blue Jay en gran parte porque detectó los atributos wing_color=blue y has_black_stripes=yes, los cuales son característicos de esa especie." Durante la inferencia en una instancia concreta, se pueden listar los atributos más influyentes presentes en esa imagen. Muchos de estos atributos el modelo los recibe ya (si usamos los verdaderos), pero si se estuvieran prediciendo internamente (concept bottleneck), igual podemos tomar los valores predichos. Otra técnica formal es entrenar un modelo interpretable surrogate usando los atributos. Por ejemplo, se puede ajustar un árbol de decisión o un conjunto de reglas lógico sobre los vectores de atributos para aproximar las predicciones del modelo complejo. Los árboles de decisión producen reglas if-then fácilmente entendibles (ejemplo de interpretabilidad global: "si wing_color = blue y bill_shape = short entonces especie = Blue Jay"), que ayudan a comprender qué combinaciones de características llevan a ciertas clasificaciones. Podemos usar los datos de entrenamiento (atributos -> predicción modelo) para inducir dicho árbol. Si el árbol logra alta fidelidad, significaría que el modelo básicamente tomó decisiones similares a esas reglas. Este método es similar a LIME pero a nivel global y solo sobre los atributos, y proporciona una explicación simbólica compacta. Alternativamente, podríamos extraer reglas por clase inspeccionando qué atributos son casi siempre verdaderos para las imágenes clasificadas como esa clase y ausentes en otras (métodos de inducción de reglas o asociación). Por ejemplo, podemos descubrir reglas como "si wing_pattern = spotted y eye_color = red, entonces Speckled Hawk" si resulta consistente en los datos. Una ventaja de disponer de atributos explícitos es que las explicaciones pueden darse en lenguaje natural estructurado. Es factible crear plantillas del estilo: "Esta ave fue clasificada como [Predicción] porque tiene [atributo1], [atributo2] y [atributo3], lo cual coincide con las características distintivas de esa especie." Por ejemplo: "Predicción: American Goldfinch. Razón: el modelo detectó plumaje amarillo brillante y alas negras con marcas blancas, atributos característicos de un jilguero americano." Estas explicaciones combinan lo visual con lo semántico: los atributos detectados pueden provenir de la imagen pero se presentan en palabras comprensibles. Si se dispone de los atributos originales de la imagen, básicamente se está reusando la etiqueta humana en la explicación, lo cual garantiza interpretabilidad (aunque en cierto sentido trivial si el humano ya lo anotó). Pero incluso si los atributos son predichos automáticamente, presentarlos al usuario valida que el modelo "vio" las mismas propiedades que un experto usaría para identificar la especie. En el contexto de nuestro modelo, para dar explicaciones simbólicas por instancia se puede proceder así: tomar el vector de atributos de la imagen (ya sea el real o el que predice internamente el modelo), y destacar aquellos que más aumentan la probabilidad de la clase predicha. Por ejemplo, usando técnicas tipo SHAP o importancias de permutation (ver siguiente sección), podemos calcular la contribución de cada atributo en la predicción específica. Luego formamos una frase con los 2-3 atributos positivos principales. Adicionalmente, podríamos mencionar algún atributo ausente que habría sugerido otra especie (p. ej., "carece de parches rojos en la cabeza, descartando el Woodpecker"). Esto da una explicación contrastiva ligera. Por último, si implementamos el modelo como un Concept Bottleneck, la explicación es intrínseca: el modelo primero estima los conceptos (atributos) y luego la clase, por lo que simplemente reportando qué conceptos fueron activados obtenemos la justificación. Estudios han mostrado que estos modelos permiten incluso intervención humana: si el modelo se equivoca en la predicción, a veces corregir manualmente un atributo predicho mal puede arreglar la clasificación
github.com
, lo que refuerza la idea de que las decisiones están efectivamente basadas en esos atributos claros.
4.3 Otros métodos y herramientas de explicabilidad
Además de Grad-CAM, LIME y las reglas basadas en atributos, existen otros enfoques útiles para entender y validar las decisiones del modelo:
SHAP (SHapley Additive exPlanations) – SHAP asigna a cada característica un valor de Shapley, basado en teoría de juegos, que representa su contribución a la diferencia entre la predicción del modelo y la media base
datacamp.com
. Aplicado a nuestro modelo, podríamos utilizar SHAP para obtener importancias tanto de píxeles/regiones (agrupándolos quizá en superpíxeles) como de atributos tabulares. En particular, SHAP es muy práctico para los atributos: nos daría una puntuación para cada atributo indicando cuánto elevó o disminuyó la probabilidad de cada clase en esa instancia. Por ejemplo, SHAP podría cuantificar que wing_color_blue aportó +0.20 a la probabilidad de Blue Jay, mientras que wing_color_blue aportaría -0.15 a la probabilidad de Cardinal (porque contradice el rojo del cardinal). Estas explicaciones son similares a las obtenidas por LIME pero con sólidas garantías teóricas de equidad en el reparto de importancia
datacamp.com
. Existen implementaciones (como la librería SHAP en Python) que soportan modelos de deep learning y tabulares, proporcionando visualizaciones como gráficos de barras de importancia o gráficos de dependencia. SHAP nos ayudaría a validar globalmente qué atributos son más utilizados por el modelo (promediando valores absolutos) y localmente entender cada predicción.
Supervisión de Atención/Partes – Si en lugar de (o adicional a) atributos, el modelo usara anotaciones de partes (por ej. localizar la cabeza, alas, cola), se podría incorporar un módulo de atención o detección de partes. Esto haría el modelo más interpretable porque podríamos ver dónde miró para cada parte. Por ejemplo, algunos modelos fine-grained detectan primero partes del ave (cabeza, alas) y luego clasifican; esas partes detectadas sirven como explicaciones: "identificó que el ave tiene la cabeza roja y el ala con barras negras". En nuestro caso, podríamos entrenar sub-modelos auxiliares para predecir la presencia de cada atributo localizando la región correspondiente. Herramientas como Grad-CAM también pueden aplicarse a salidas intermedias (por ejemplo, al neuron de "blue wing" en la red de atributos) para ver si la CNN realmente mira a las alas cuando activa ese atributo – esto combina explicabilidad visual y simbólica.
Ejemplos prototípicos – Otra técnica explicativa es proporcionar ejemplos similares o prototipos. Dado que tenemos un conjunto de entrenamiento etiquetado, podemos al predecir una imagen recuperar las imágenes de entrenamiento más cercanas en el espacio de características conjunto. Si, por ejemplo, el modelo clasifica una imagen como Western Meadowlark, podríamos mostrar al usuario otras imágenes de Western Meadowlark del set que el modelo considere parecidas (quizá porque también tienen pecho amarillo y una banda negra en el cuello). Esto ayuda a contextualizar la predicción: "se predijo X porque la imagen se parece a estos ejemplos conocidos de X". Incluso se pueden mostrar comparaciones con la especie más probable alternativa, mostrando diferencias. Esta técnica no explica atributos específicos, pero proporciona una intuición basada en casos.
Contra-factuales – Generar explicaciones contrafactuales consiste en responder: "¿qué tendría que cambiar en esta entrada para que el modelo diera otra clasificación?". Con atributos, esto es relativamente fácil: por ejemplo, "si esta ave no tuviera las alas azules (supongamos fueran marrones), el modelo la clasificaría como un Sparrow en lugar de Blue Jay*". Esa es una explicación contrafactual que señala la influencia de "alas azules". Podemos automatizar esto probando a modificar atributos clave y viendo cómo cambia la predicción. En imágenes puras es más difícil (implica editar la imagen), pero con atributos es factible y muy interpretable ("cambia X a Y, modelo output cambia a ..."). Esto puede dar confianza de que el modelo captura relaciones coherentes.
Resumen de métodos de explicabilidad: A modo de síntesis, la siguiente tabla resume las opciones mencionadas, indicando el tipo de explicación que brindan:
Método	Descripción breve	Modalidad principal
Grad-CAM	Mapa de calor sobre la imagen que resalta las regiones relevantes utilizadas por la CNN para una clase dada (p. ej., señala partes del ave importantes para la predicción).	Visual (imagen)
LIME (imágenes)	Modelo lineal local sobre superpíxeles de la imagen; indica qué segmentos de la imagen contribuyen positiva o negativamente a la predicción actual.	Visual (imagen)
Árbol de Decisión (Reglas)	Aproxima el modelo usando un árbol sobre los atributos; produce reglas tipo if-then comprensibles (ej: si ala=azul y pico=corto entonces Blue Jay), revelando la lógica basada en atributos.	Simbólico (atributos)
SHAP (Valores Shapley)	Asigna a cada característica (atributo o región) una importancia para la predicción, explicando cuánto influyó en comparación a la media. Útil para identificar los atributos más determinantes en cada caso.	Ambos (imagen/atributos)
Concept Bottleneck	(Arquitectura interpretable) El modelo predice primero atributos y luego la clase; la explicación es inmediata: la lista de atributos predichos actúa como justificación de la clase. Permite intervención humana sobre atributos.	Simbólico (atributos)
Cada método tiene su rol: Grad-CAM y LIME ofrecen transparencia visual, mientras que las reglas, SHAP o el uso de conceptos dan transparencia lógica/simbólica. Idealmente, combinaríamos ambos tipos para una explicación completa. Por ejemplo, para una predicción podríamos mostrar un mapa Grad-CAM indicando dónde miró el modelo en la foto del pájaro, junto con una frase basada en atributos explicando qué características notó (y quizás una regla simplificada). De esta forma, usuarios tanto técnicos como no técnicos (ornitólogos, por ejemplo) podrían validar la predicción: verían que el modelo observó las partes correctas del ave y mencionó las cualidades (color, forma) esperadas para esa especie, aumentando la confianza en la decisión.
5. Herramientas y Bibliotecas para la Implementación
Para implementar el enfoque propuesto, podemos apoyarnos en diversas bibliotecas de deep learning y XAI (eXplainable AI) disponibles:
Frameworks de Deep Learning: Tanto PyTorch como TensorFlow/Keras son adecuados para construir la arquitectura multimodal. En PyTorch, se puede definir fácilmente un modelo con dos ramas (usando la clase torch.nn.Module personalizada) que tome dos entradas (imagen y atributos) y las combine. PyTorch Lightning también facilita el entrenamiento organizado. En Keras (TensorFlow), la Functional API permite combinar múltiples entradas de forma declarativa
pyimagesearch.com
pyimagesearch.com
. Por ejemplo, se definiría una entrada input_image (pasada por una base CNN pre-entrenada como ResNet50, posiblemente congelando capas inicialmente) y otra input_attrs (pasada por capas Dense). Luego se usa keras.layers.Concatenate() para fusionarlas y construir el resto del modelo. Ambas frameworks soportan entrenar con múltiples inputs naturalmente.
Modelos pre-entrenados: Dado el tamaño relativamente modesto de CUB (≈6k imágenes de entrenamiento), es recomendable usar un modelo de imagen pre-entrenado en ImageNet y luego afinarlo (transfer learning). En PyTorch, torchvision.models provee muchos arquitecturas CNN pre-entrenadas (ResNet, DenseNet, etc.). Podemos cargar ResNet50, reemplazar su capa final por una identidad para obtener features, y usar esas features concatenadas con los atributos. En Keras, tf.keras.applications igualmente ofrece modelos pre-entrenados listos para usar. Esto acelerará la convergencia y probablemente mejore la precisión.
Procesamiento de datos: Para manejar la lectura de imágenes y atributos, bibliotecas como pandas pueden ayudar con el archivo de atributos (cargar el CSV/TXT de atributos en dataframes, pivotearlo a formato ancho por imagen). OpenCV o PIL junto con frameworks mencionados para cargar las imágenes. Además, se pueden usar las utilidades de dataset de PyTorch (torch.utils.data.Dataset) o TensorFlow (tf.data.Dataset) para crear loaders que entreguen tuplas (imagen, atributos, etiqueta).
Explicabilidad Visual:
Grad-CAM: Existe la librería pytorch-grad-cam (de Jacob Gil) que simplifica obtener Grad-CAM en modelos PyTorch con apenas unas líneas, incluso soporta modelos con múltiples inputs (tomando gradientes respecto a la imagen). En TensorFlow/Keras, hay implementaciones oficiales y oficiosas; por ejemplo tf-explain o simplemente calcular gradientes manualmente con GradientTape. Otra opción es Captum (de Facebook) para PyTorch, que ofrece captum.attr.LayerGradCam y métodos para Integrated Gradients, etc.
LIME: La librería lime de Python soporta explicación de imágenes. Sólo requiere proporcionar una función lambda que dado una imagen (o lote) devuelva la predicción del modelo; luego lime_image.LimeImageExplainer segmenta la imagen y genera la explicación. Nos dará las máscaras de superpíxeles más importantes que podemos sobreponer en la imagen para visualización.
Integrated Gradients: Integrated Gradients está implementado en Captum (PyTorch) y en tf-explain/Alibi (Keras). Por ejemplo, Captum tiene IntegratedGradients que se usa pasando la imagen y una imagen baseline (negra) para obtener la atribución de cada pixel.
Herramientas de visualización: Matplotlib puede servir para mostrar las imágenes con los heatmaps encima. OpenCV también, pero matplotlib integrará bien en notebooks. Para Grad-CAM, se suele aplicar un colormap tipo jet semi-transparente sobre la imagen original.
Explicabilidad Simbólica:
Extracción de reglas: scikit-learn proporciona implementaciones de DecisionTreeClassifier que podemos entrenar sobre los atributos. También librerías como sklearn.tree.export_text permiten volcar la estructura del árbol en texto legible. Si se desea reglas más compactas que un único árbol (p.ej., reglas de asociación), se podría usar librerías de minería de datos (como mlxtend.frequent_patterns para reglas apriori) aunque tal vez sea overkill; un árbol ya resume bastante.
SHAP: La librería SHAP (pip install shap) funciona con modelos tipo scikit-learn y también tiene un DeepExplainer para redes neuronales. Para nuestra mezcla, podríamos aplicar KernelExplainer tratando el modelo como caja negra (aunque es lento si la dimensión es grande) o usar DeepExplainer con la parte de atributos y parte de imagen por separado. Una idea: obtener SHAP values solo del submodelo de atributos (más interpretable) alimentándolo con los atributos reales, para ver su contribución. SHAP también tiene visualizaciones listas (bar charts, force plots).
Concept bottleneck interpretability: Si implementamos el CBM, la salida intermedia de conceptos se puede inspeccionar. Si usamos PyTorch, simplemente tomando el vector de predicciones de atributos podemos interpretarlo antes de la capa final. No requiere librería especial, más allá de formatear esa info para el usuario.
Entrenamiento y evaluación:
Entrenamiento: Además de los frameworks base, se puede usar PyTorch Lightning para estructurar el entrenamiento (separa lógica de forward, training step, etc., útil para multimodal), o TensorFlow ModelCheckpoint y EarlyStopping callbacks para guardar el mejor modelo. Dado que es multimodal, conviene monitorear la loss total y accuracy en validación.
Hiperparámetros: Probablemente necesaria cierta experimentación con la proporción de neuronas en la rama de atributos vs visual. Una herramienta como Weights & Biases o TensorBoard puede ayudar a hacer tracking de experimentos.
Computación: Como son ~12k imágenes, entrenar en GPU es recomendable. La arquitectura propuesta no es excesivamente grande (ResNet50 + algunas capas densas), por lo que una sola GPU moderna podría entrenar en minutos-horas.
Evaluación de Explicaciones:
Para visualizar explicaciones de forma interactiva, herramientas como Plotly Dash o paneles tipo TensorBoard plugins podrían ser útiles. Sin embargo, en un entorno de investigación, probablemente generemos las explicaciones offline y las analicemos manualmente.
Si se quisiera integrar en una aplicación, se podría usar Gradio o Streamlit para hacer una demo web donde un usuario sube una foto de ave y el sistema muestra la predicción con un Grad-CAM overlay y algunas frases explicativas.
En resumen, el ecosistema Python ofrece todo lo necesario: PyTorch/TensorFlow para la red multimodal, pandas/sklearn para manejar atributos y explicaciones simbólicas, y librerías especializadas como lime, shap, captum para las explicaciones avanzadas. La recomendación es apoyarse en estas bibliotecas en lugar de reinventar métodos, para asegurar resultados confiables. Por ejemplo, usar shap para computar importancias de atributos garantiza que estamos siguiendo un método bien establecido y no una heurística ad-hoc.
6. Evaluación del Rendimiento y Calidad Explicativa
Finalmente, diseñaremos cómo evaluar tanto la precisión del modelo como la utilidad de sus explicaciones: Rendimiento del modelo: Dado que es un problema de clasificación multiclase (200 especies), el principal métrico será la exactitud (accuracy) sobre el conjunto de prueba. Conviene reportar también la accuracy top-5, ya que en fine-grained a veces el modelo falla por confundir especies muy cercanas; una top-5 alta indicaría que la verdadera especie suele estar entre las 5 con mayor puntaje. Además, podemos calcular una matriz de confusión para identificar patrones de error: ¿confunde consistentemente cierto gorrión con otro de plumaje similar? Esto podría indicar qué atributos o rasgos está ignorando. Dado que integramos atributos, sería interesante comparar la performance con un modelo solo imágenes (CNN pura) y solo atributos (por ejemplo, un clasificador entrenado únicamente con el vector de 312 atributos). Es esperable que la combinación supere a cada uno por separado; verificarlo cuantitativamente validará la utilidad de la multimodalidad. También podemos medir la pérdida de clasificación (cross-entropy) y observar su convergencia durante el entrenamiento para asegurar que el modelo no está sobreajustando (idealmente la pérdida de validación bajará similar a la de entrenamiento y se estabilizará). En caso de haber entrenado un modelo de cuello de botella de conceptos, evaluaremos también la exactitud en la predicción de atributos (qué porcentaje de los 312 atributos predice correctamente por imagen). Si este accuracy de atributos es alto, entonces el modelo tiene una buena comprensión de los conceptos, lo que probablemente conduce a mejor clasificación. Si es bajo en algunos atributos específicos, podríamos identificar cuáles son difíciles de detectar visualmente (por ejemplo, tal vez "color de las patas" es difícil si en muchas fotos no se ven bien las patas). Calidad de las explicaciones: Evaluar explicaciones es más subjetivo que evaluar precisión, pero hay estrategias tanto cualitativas como cuantitativas:
Evaluación cualitativa (inspección humana): Reunir un muestreo de imágenes de prueba y para cada una mostrar la predicción del modelo junto con sus explicaciones (mapas Grad-CAM, atributos destacados, etc.). Estas explicaciones pueden ser revisadas por expertos en aves o los mismos desarrolladores para juzgar si son plausibles y coherentes con el conocimiento ornitológico. Por ejemplo, si el modelo dice "alas azules y pecho blanco" para identificar un Eastern Bluebird, y efectivamente esas son características distintivas de esa especie, la explicación se considera buena. Si en cambio destaca algo irrelevante (fondo verde, rama en la que posa el ave) como razón, entonces la explicación revela un posible problema (modelo sobreajustado al contexto). Esta revisión cualitativa ayuda a identificar fallos de razonamiento del modelo que no son obvios solo mirando el accuracy.
Localización de partes relevantes: Dado que CUB proporciona anotaciones de partes del cuerpo (coordenadas del pico, ojo, ala, cola, etc.), podemos cuantificar si los mapas de calor como Grad-CAM se solapan con las partes relevantes para la clasificación. Por ejemplo, para cada imagen podríamos comprobar si la región de máxima intensidad en Grad-CAM cae dentro del cuadro delimitador del ave o de una parte particular (digamos la cabeza o alas). Una buena explicación visual debería estar focalizada en el pájaro, no en el fondo. Podemos calcular la fracción de imágenes donde el peak del mapa cae sobre el objeto correcto ("pointing game metric"). Asimismo, si sabemos que cierta parte es clave (p. ej. la mancha en la cabeza distingue dos especies), esperaríamos que el mapa destaque la cabeza; las anotaciones permitirían verificarlo (medir IoU – intersección sobre unión – entre el heatmap binarizado y la máscara de la cabeza, por ejemplo).
Fidelidad de las explicaciones: Para métodos como LIME o los árboles surrogate, podemos medir qué tan bien aproximan el modelo original. LIME ya provee una medida de ajuste local (R^2 del modelo lineal local). Para un árbol global, podemos computar el accuracy del árbol en predecir las salidas del modelo complejo en un conjunto de datos; un porcentaje alto significa que las reglas del árbol casi emulan al modelo. No obstante, demasiada fidelidad podría llevar a un árbol enorme poco interpretable, hay que balancear tamaño vs fidelidad.
Experimentos contrafactuales: Podemos probar a modificar entradas de forma controlada para validar la explicación. Por ejemplo, si la explicación dijo "alas azules -> Blue Jay", podemos editar esa imagen (Photoshop, manual) para cambiar el color azul a otro color y ver si el modelo deja de predecir Blue Jay. Esto obviamente es manual y difícil a gran escala, pero se puede simular con otras imágenes: tomar una imagen de Blue Jay y otra de un ave similar con alas diferentes, e intercambiar atributos o regiones, comprobando si el modelo cambia su predicción acorde. Si el modelo es interpretable y correcto, debería comportarse de forma consistente con los atributos (ej., si alimentamos los atributos de un Cardinal junto con la foto de un Blue Jay, ¿qué hace? Idealmente debería dudar o predecir otra cosa).
Métricas de coherencia global: Si disponemos de muchas explicaciones, podríamos evaluar si siguen un patrón lógico. Por ejemplo, para todas las predicciones de Blue Jay en el test, ¿cuántas veces las explicaciones mencionan "blue" o resaltan la zona azul? Debería ser frecuente. Si encontramos explicaciones dispares para la misma clase (en un caso dice alas azules, en otro dice cola larga, etc.), podría ser que el modelo use múltiples vías, lo cual puede ser válido si la especie tiene varios rasgos, pero también podría indicar inconsistencia. Una medida llamada consistencia de explicaciones busca que casos similares tengan explicaciones similares. Herramientas como TCAV (Testing with Concept Activation Vectors) pueden evaluar si la sensibilidad a un concepto (p. ej. "azul") es alta para la clase que debería ser (Blue Jay) y no para otras.
Satisfacción del usuario: Si este sistema fuera para uso de expertos o ciudadanos científicos, una evaluación importante es la satisfacción y confianza del usuario en las explicaciones. Esto se suele medir mediante encuestas o estudios de usuario: mostrar predicciones explicadas vs no explicadas y preguntar cuánta confianza les genera, o si pueden detectar errores deliberados. Dado que nuestro contexto es más técnico, nos centramos en métricas automáticas, pero al final si el objetivo es "explicaciones comprensibles", la prueba de fuego es que un humano las entienda y las considere justificadas.
En conclusión, para la precisión seguiremos las métricas estándar de clasificación (accuracy global y por clase, etc.), asegurando que el modelo multimodal supera baselines. Para la explicabilidad, aplicaremos una batería de métodos (Grad-CAM, LIME, SHAP, reglas) y evaluaremos su coherencia con el dominio del problema. Un modelo explicable ideal permitirá afirmar con confianza: "el clasificador distingue correctamente las especies usando los mismos rasgos que usaría un ornitólogo humano" – y nuestras evaluaciones visuales/simbólicas deben corroborar esto. Si encontramos discrepancias (por ejemplo, el modelo se fija en el fondo o en un atributo irrelevante), tendremos oportunidad de depurar el modelo (ajustar el entrenamiento, agregar regularización, o incluso incorporar esas observaciones como retroalimentación). La capacidad de evaluar y confiar en las explicaciones es justamente lo que diferencia este enfoque de una simple caja negra: no solo importa el porcentaje de aciertos, sino verificar cómo se logran esos aciertos, garantizando un sistema más transparente y fiable.