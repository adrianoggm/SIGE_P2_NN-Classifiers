import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# Para Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Asume que config.py define DEVICE ('cuda' o 'cpu') y EPOCHS
from config import DEVICE, EPOCHS

class MultiModalResNet(nn.Module):
    '''
    Modelo multimodal que combina caracteristicas visuales (imagen)
    y atributos estructurados (vector de atributos) para clasificacion.
    '''
    def __init__(self, num_classes: int, attr_dim: int, attr_embed_dim: int = 256):
        super().__init__()
        self.image_model = models.resnet18(pretrained=True)
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()

        self.attr_fc = nn.Sequential(
            nn.Linear(attr_dim, attr_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + attr_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        img_feats = self.image_model(images)
        attr_feats = self.attr_fc(attrs)
        fused = torch.cat([img_feats, attr_feats], dim=1)
        out = self.classifier(fused)
        return out


def train_model(model, train_loader, val_loader, learning_rate, optimizer_name, save_best=True):
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    best_val_acc = 0.0
    model.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, attrs, labels in train_loader:
            images, attrs, labels = images.to(DEVICE), attrs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, attrs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, attrs, labels in val_loader:
                images, attrs, labels = images.to(DEVICE), attrs.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, attrs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_accuracy = correct / total * 100

        if save_best and val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model_multimodal.pth')

        print(f'Epoch {epoch+1}/{EPOCHS} - Val Acc: {val_accuracy:.2f}%')

    return best_val_acc


def hyperparameter_tuning(train_dataset, val_dataset, full_dataset, attr_dim):
    from itertools import product
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'sgd']
    }
    param_combinations = list(product(*param_grid.values()))
    num_classes = len(full_dataset.class_to_idx)

    best_config, best_accuracy = None, 0.0
    for lr, batch_size, opt in param_combinations:
        print(f'\nProbando: lr={lr}, batch_size={batch_size}, optimizer={opt}')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        model = MultiModalResNet(num_classes, attr_dim).to(DEVICE)
        val_acc = train_model(model, train_loader, val_loader, learning_rate=lr, optimizer_name=opt)
        print(f'Validación: {val_acc:.2f}%')
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': opt}

    print(f'\nMejor configuración: {best_config} con Val Acc: {best_accuracy:.2f}%')
    best_train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
    best_val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'])
    final_model = MultiModalResNet(len(full_dataset.class_to_idx), attr_dim).to(DEVICE)
    train_model(final_model, best_train_loader, best_val_loader,
                best_config['learning_rate'], best_config['optimizer'])

    return best_config


def generate_gradcam(model, img_tensor, attr_tensor, target_class=None):
    model.eval()
    cam = GradCAM(model=model,
                  target_layers=[model.image_model.layer4],
                  use_cuda=(DEVICE == 'cuda'))

    if target_class is None:
        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0).to(DEVICE), attr_tensor.unsqueeze(0).to(DEVICE))
            target_class = outputs.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=[img_tensor.to(DEVICE), attr_tensor.to(DEVICE)], targets=targets)[0]

    rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_img, target_class

# Ejemplo de uso:
#   from dataset import CUBMultimodalDataset
#   train_ds = CUBMultimodalDataset(...)
#   val_ds = CUBMultimodalDataset(...)
#   best_cfg = hyperparameter_tuning(train_ds, val_ds, train_ds, attr_dim=312)
#   model = MultiModalResNet(num_classes=200, attr_dim=312)
#   model.load_state_dict(torch.load('best_model_multimodal.pth'))
#   img, attrs, label = val_ds[0]
#   cam_img, pred_class = generate_gradcam(model, img, attrs)
#   # Mostrar cam_img con PIL o matplotlib
