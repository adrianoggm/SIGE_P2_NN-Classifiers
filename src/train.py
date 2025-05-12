import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from itertools import product
from config import DEVICE, EPOCHS

from src.customCNN import CustomCNN

def get_model(num_classes, model_type='resnet'):
    if model_type == 'resnet':
        # Cargamos modelo preentrenado
        model = models.resnet18(pretrained=True)

        # Congelamos todas las capas excepto las últimas
        for param in model.parameters():
            param.requires_grad = False

        # Añadimos nuevas capas
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),  # Capa FC intermedia
            nn.ReLU(),  # Función de activación ReLU
            nn.Dropout(0.5),  # Dropout para evitar overfitting
            nn.Linear(512, num_classes)  # Capa final con salida de clases
        )
    elif model_type == 'custom':
        model = CustomCNN(num_classes)
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")
    return model.to(DEVICE)


def train_model(model, train_loader, val_loader, learning_rate, optimizer_name, save_best=True):
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # Calcular la pérdida media en el entrenamiento
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_accuracy = correct / total * 100


        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")


        if save_best and val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    return best_val_acc

def hyperparameter_tuning(train_dataset, val_dataset, full_dataset):
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'sgd']
    }

    param_combinations = list(product(*param_grid.values()))
    num_classes = len(full_dataset.class_to_idx)

    best_config = None
    best_accuracy = 0.0

    for lr, batch_size, opt in param_combinations:
        print(f"\nProbando: lr={lr}, batch_size={batch_size}, optimizer={opt}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = get_model(num_classes)
        val_acc = train_model(model, train_loader, val_loader, learning_rate=lr, optimizer_name=opt)

        print(f"Validación: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'optimizer': opt
            }

    print(f"\nMejor configuración encontrada: {best_config}")
    print(f"Precisión en validación: {best_accuracy:.2f}%")

    # 🔁 Entrenar nuevamente usando los mejores hiperparámetros
    print("\n🏁 Entrenando modelo final con mejores hiperparámetros...")
    best_train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
    best_val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'])
    final_model = get_model(num_classes)
    train_model(final_model, best_train_loader, best_val_loader, best_config['learning_rate'], best_config['optimizer'])

    return best_config
