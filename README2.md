# SIGE_P2_NN-Classifiers: Información de entrega

## Entrega
A continuación, se detallan los ficheros entregados:

- *main.py:* Fichero para el modelo básico.
    - *train.py:* Módulo para los métodos de entrenamiento del modelo básico.
    - *customCNN.py*: Red convolucional propia realizada para el modelo básico.
    - *data_loader.py*: Módulo para cargar los datos de entrada (imágenes) y se realizan las técnicas para mejorar el aprendizaje (redimensionado, data augmentation)
- *main_explicable.py:* Fichero para el modelo multimodal y explicable
    - *explicable_train.py:* Módulo para los métodos de entrenamiento del modelo multimodal y explicable.
    - *explicable_data_loader.py:* Módulo para cargar los datos de entrada (imágenes y atributos) y se realizan las técnicas para mejorar el aprendizaje (redimensionado, data augmentation)
    - *utils.py* y visualization.py: Módulo de utilizades para el preprocesamiento y la visualización de los datos.
- *config.py:* Fichero con variables de configuración
- *memoria.pdf:* Documentación de la práctica realizada. Si bien su extensión supera las recomendaciones iniciales al incorporar numerosos análisis, gráficos y tablas (junto con la portada, el índice y la bibliografía), hemos optado por mantener este formato único para garantizar una presentación coherente y estructurada del trabajo. Contiene: 
    - Desarrollo detallado de cada apartado del guión
    - Visualizaciones y resultados completos
- *README.md:* Documentación para el repositorio de GitHub

## 📝 CheckList
- **Utilización de subconjunto de 20 categorías** ✔️
- **Análisis exploratorio** ✔️
- **Particionamiento de datos** ✔️
- **Clasificación multiclase** ✔️
- **Ajuste de hiperparámetros, topología de la red, función de coste y optimizador** ✔️
- **Aplicación y estudio de 2 técnicas para mejora del aprendizaje** ✔️. Transformación estándar para redimensionar las imágenes a 224x224 y Data Augmentation con varias operaciones.
- **Ampliación de la solución básica para las 200 clases** ✔️
- **Uso de Weights And Biases** ✔️
- **Aprendizaje multimodal (imágenes + datos tabulares + descripciones en lenguaje natural u otros)** ✔️
- **Aprendizaje con coste variable (por ejemplo, modificación de la función de pérdida2)** ❌
- **Ajuste automático de hiperparámetros** ✔️. Se ha utilizado OPTUNA
- **Métodos de tipo ensemble e híbridos** ❌
- **Transferencia de aprendizaje o fine tuning** ✔️
- **Explicabilidad del modelo** ✔️. Se ha utilizado Lime y GradCam
