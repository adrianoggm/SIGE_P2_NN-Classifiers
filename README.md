# SIGE_P2_NN-Classifiers

## 📚 Overview

This repository provides a modular solution for **bird classification** using images at two different resolutions (**20x20** and **200x200**), developed for the SIGE course. The project supports both classic and explainable multimodal training (image + attributes), as well as advanced visualization and hyperparameter search.

---

## 🗂️ Project Structure

- **config.py** — Global configuration (paths, training parameters, device).
- **src/data_loader.py** — Classic data loading and preprocessing (transforms, splits, DataLoaders).
- **src/explicable_data_loader.py** — Multimodal data loading (image + attributes).
- **src/visualization.py** — Visualization tools for images and class distribution.
- **src/explicable_visualization.py** — Visualization of examples and attributes in explainable mode.
- **src/train.py** — Model definitions, classic training, validation, and tuning.
- **src/explicable_train.py** — Training and tuning for explainable multimodal models.
- **main.py** — Classic workflow: loading, visualization, training, and evaluation.
- **main_explicable.py** — Explainable workflow: multimodal training and visualization.
- **gradcam.py** — Grad-CAM and LIME explanations visualization for dataset examples.

---
## 📝 Checklist

- Use of a subset of 20 categories ✔️
- Exploratory data analysis ✔️
- Data partitioning ✔️
- Multiclass classification ✔️
- Hyperparameter tuning, network topology, loss function, and optimizer ✔️
- Application and study of 2 techniques to improve learning ✔️  
  (Standard transformation to resize images to 224x224 and Data Augmentation with several operations)
- Extension of the basic solution to all 200 classes ✔️
- Use of Weights & Biases (wandb) ✔️
- Multimodal learning (images + tabular data + natural language descriptions or others) ✔️
- Variable cost learning (e.g., loss function modification) ❌
- Automatic hyperparameter tuning ✔️ (OPTUNA used)
- Ensemble and hybrid methods ❌
- Transfer learning or fine-tuning ✔️
- Model explainability ✔️ (LIME and GradCAM used)
---
## ⚙️ Installation

1. **Clone the repository and enter the folder:**
   ```bash
   git clone https://github.com/adrianoggm/SIGE_P2_NN-Classifiers.git
   cd SIGE_P2_NN-Classifiers
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** The `requirements.txt` file includes dependencies such as `torch`, `torchvision`, `matplotlib`, `wandb`, `optuna`, `lime`, `pytorch-grad-cam`, etc.

---

## 🛠️ Configuration

Edit `config.py` to adjust:

- **Dataset paths:**  
  - `DATA_DIR_X20` — Folder with 20x20 images  
  - `DATA_DIR_X200` — Folder with 200x200 images  
  - Paths to attribute files for explainable mode

- **Training parameters:**  
  - `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`

- **Device:**  
  - Uses AMD GPU via DirectML if available.

- **Environment variables:**  
  - `MAIN_DATASET` (`x20` or `x200`) to select the main dataset.
  - `FINE_TUNE` to enable hyperparameter search.

---

## 🚀 Quick Start

### Classic training (images only)

```bash
python main.py
```

- Loads data, shows examples and distributions, trains a model (ResNet/EfficientNet/CustomCNN).
- Saves the best model as `best_model.pth`.

### Explainable training (image + attributes)

```bash
python main_explicable.py
```

- Loads images and attribute vectors.
- Trains a multimodal model (`MultiModalResNet`).
- Supports hyperparameter tuning and visualization of examples and attributes.
- Saves the best model as `best_model_multimodal.pth`.

### Explanations visualization (Grad-CAM and LIME)

```bash
python gradcam.py
```

- Shows visual explanations for predictions of the multimodal model.

---

## 🧪 Data Flow & Preprocessing

- **Transformations:**  
  - Resizing, normalization, augmentations (flip, rotation, zoom).
- **Split:**  
  - 80% training, 20% validation.
- **Augmentation:**  
  - Concatenates original and augmented data for robustness.

---

## 📊 Visualization

- **src/visualization.py:**  
  - Shows image grids, pixel histograms, class distribution.
- **src/explicable_visualization.py:**  
  - Visualizes multimodal examples and attributes.

---

## 🧠 Models & Training

- **Supported models:**  
  - ResNet50 (selective fine-tuning)
  - EfficientNet-B4
  - CustomCNN (simple)
  - MultiModalResNet (image + attributes)

- **Training:**  
  - Loop with scheduler, early stopping, logging with `wandb`.
  - Automatic saving of the best model.

- **Tuning:**  
  - Grid search and Optuna for hyperparameters (`learning_rate`, `batch_size`, `optimizer`).

---

## 🧩 Explainability

- **Multimodal models:**  
  - Fuse image and attributes for improved interpretation.
- **Visual explanations:**  
  - Grad-CAM and LIME to understand which parts of the image and attributes influence predictions.

---

## 📈 Example Results

- **Training:**  
  - Shows loss and accuracy per epoch.
- **Visualization:**  
  - Examples of images, distributions, and visual explanations.

---

## 🤝 Contributions

Pull requests and suggestions are welcome. Please open an issue to discuss major changes.

---

## 📄 License

This project is distributed under the MIT license.

---

**Developed for the SIGE course.**