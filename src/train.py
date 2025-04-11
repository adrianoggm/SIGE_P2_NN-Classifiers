import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from config import DEVICE, EPOCHS, LEARNING_RATE

def get_model(num_classes):
    """
    Retorna un modelo ResNet18 preentrenado, ajustado a num_classes.
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def train_model(model, train_loader, val_loader):
    """
    Ejecuta el entrenamiento y validación del modelo.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        
        # Entrenamiento
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")
        
        # Validación
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
        print(f"Validation Accuracy: {val_accuracy:.2f}%\n")
