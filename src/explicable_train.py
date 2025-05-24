from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import wandb
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import DEVICE, EPOCHS
from src.explicable_data_loader import custom_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MultiModalResNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 attr_dim: int,
                 attr_embed_dim: int = 256,
                 dropout_rate: float = 0.5):
        super().__init__()
        # 1) Cargo ResNet-50 preentrenada
        self.image_model = models.resnet50(pretrained=True)
        # 2) Congelo TODO el backbone
        for param in self.image_model.parameters():
            param.requires_grad = False

        # 3) Descongelo sólo layer4 (último bloque residual)
        for name, param in self.image_model.named_parameters():
            if name.startswith("layer4"):
                param.requires_grad = True

        # 4) Sustituyo la cabeza original por Identity, para extraer sólo features
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()

        # 5) Módulo de atributos (siempre entrenable)
        self.attr_fc = nn.Sequential(
            nn.Linear(attr_dim, attr_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # 6) Cabeza multimodal: fusión de imagen + atributos
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + attr_embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, images: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        # Extraigo features de la imagen (ResNet-50 hasta layer4)
        img_feats = self.image_model(images)          # requiere grad solo layer4
        # Embedding de atributos
        attr_feats = self.attr_fc(attrs)              # todo entrenable
        # Fusiono y clasifico
        fused = torch.cat([img_feats, attr_feats], dim=1)
        return self.classifier(fused)   


def train_model(model, train_loader, val_loader,
                learning_rate, optimizer_name,
                save_best=True, use_wandb=False, with_htuning=False):
    if use_wandb and not with_htuning:
        wandb.init(
            project="clasificacion-explicable",
            config={
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "epochs": EPOCHS
            },
            reinit=True 
        )

    criterion = nn.CrossEntropyLoss()
    # --- optimizador ---
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    # --- scheduler ReduceLROnPlateau igual que en tu ejemplo ---
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',        # maximizamos val_accuracy
        factor=0.15,       # lr_new = lr_old * 0.15
        patience=3,        # tras 3 epochs sin mejora
        verbose=True
    )

    best_val_acc = 0.0
    lr_reduction_count = 0
    prev_lr = learning_rate

    model.to(DEVICE)
    for epoch in range(EPOCHS):
        # -------- entrenamiento --------
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
        train_loss = running_loss / len(train_loader.dataset)

        # -------- validación --------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, attrs, labels in val_loader:
                images, attrs, labels = images.to(DEVICE), attrs.to(DEVICE), labels.to(DEVICE)
                outputs = model(images, attrs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_accuracy = correct / total * 100

        # actualizar scheduler y obtener lr actual
        scheduler.step(val_accuracy)
        curr_lr = optimizer.param_groups[0]['lr']

        # comprobar si mejoró
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            lr_reduction_count = 0
            if save_best:
                torch.save(model.state_dict(), 'best_model_multimodal.pth')
        else:
            # si lr bajó sin mejora, incrementa contador
            if curr_lr < prev_lr:
                lr_reduction_count += 1
                print(f"Learning rate reduced to {curr_lr:.6f} (count: {lr_reduction_count})")
            # early-stopping tras 3 reducciones sin mejora
            if lr_reduction_count >= 3:
                print("No improvement after 3 LR reductions. Stopping training early.")
                break

        prev_lr = curr_lr

        # logging y salida por consola
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Val Acc: {val_accuracy:.2f}%  LR: {curr_lr:.6f}")
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "lr": curr_lr
            })

    if use_wandb and not with_htuning:
        wandb.log({"final_val_accuracy": best_val_acc})
        wandb.finish()

    return best_val_acc


def hyperparameter_tuning(train_dataset, val_dataset, full_dataset):
    """
    Realiza búsqueda en cuadrícula sobre learning_rate, batch_size y optimizer.
    Construye DataLoaders multimodales con custom_collate.
    """
    from itertools import product
    # Grid de parámetros
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [32, 64],
        'optimizer': ['adam', 'sgd']
    }
    combinations = list(product(*param_grid.values()))

    # Determinar número de clases
    # Intentar usar class_to_idx, si no existe usar img_folder.classes
    try:
        num_classes = len(full_dataset.class_to_idx)
    except AttributeError:
        num_classes = len(full_dataset.img_folder.classes)
    # Dimensión de atributos
    attr_dim = getattr(full_dataset, 'attr_dim', None) or getattr(full_dataset, 'image_attrs')[next(iter(full_dataset.image_attrs))].shape[0]

    best_config = None
    best_accuracy = 0.0
    for lr, batch_size, opt in combinations:
        print(f'Probando: lr={lr}, batch_size={batch_size}, optimizer={opt}')
        wandb.init(
            project="clasification-explicable",
            config={
                "learning_rate": lr,
                "batch_size": batch_size,
                "optimizer": opt,
                "epochs": EPOCHS
            },
            reinit=True
        )       
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=custom_collate)

        model = MultiModalResNet(num_classes, attr_dim).to(DEVICE)
        val_acc = train_model(model, train_loader, val_loader,
                              learning_rate=lr, optimizer_name=opt, use_wandb=True, with_htuning=True)
        
        print(f'Validación: {val_acc:.2f}%')
        wandb.log({"final_val_accuracy": val_acc})
        wandb.finish()  # 👈 Cerrar el run

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': opt}

    print(f'Mejor configuración: {best_config} con Val Acc: {best_accuracy:.2f}%')

    # Entrenar modelo final con mejores parámetros
    final_train = DataLoader(train_dataset, batch_size=best_config['batch_size'],
                             shuffle=True, collate_fn=custom_collate)
    final_val = DataLoader(val_dataset, batch_size=best_config['batch_size'],
                           shuffle=False, collate_fn=custom_collate)
    final_model = MultiModalResNet(num_classes, attr_dim).to(DEVICE)
    train_model(final_model, final_train, final_val,
                best_config['learning_rate'], best_config['optimizer'], use_wandb=True, with_htuning=True)
    
    # Guardar el modelo final
    wandb.init(project="tuning-clasificacion-explicable", name="final_model", config=best_config, use_wandb=True)
    train_model(final_model, final_train, final_val, best_config['learning_rate'], best_config['optimizer'], use_wandb=True, with_htuning=True)
    wandb.finish()

    return best_config

def generate_gradcam(model, img_tensor, attr_tensor, target_class=None):
    model.eval()
    img = img_tensor.unsqueeze(0).to(DEVICE)
    attr = attr_tensor.unsqueeze(0).to(DEVICE)

    if target_class is None:
        with torch.no_grad():
            outputs = model(img, attr)
            target_class = outputs.argmax(dim=1).item()

    class Wrapper(nn.Module):
        def __init__(self, base_model, fixed_attr):
            super().__init__()
            self.base = base_model
            self.attr = fixed_attr

        def forward(self, x):
            return self.base(x, self.attr)

    wrapper = Wrapper(model, attr)
    cam = GradCAM(model=wrapper, target_layers=[model.image_model.layer4])
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=img, targets=targets)[0]
    rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_img, target_class


def generate_lime_explanation(model, img_tensor, attr_tensor, target_class=None, num_samples=1000):
    model.eval()
    img = img_tensor.unsqueeze(0).to(DEVICE)
    attr = attr_tensor.unsqueeze(0).to(DEVICE)

    if target_class is None:
        with torch.no_grad():
            outputs = model(img, attr)
            target_class = outputs.argmax(dim=1).item()

    class Wrapper(nn.Module):
        def __init__(self, base_model, fixed_attr):
            super().__init__()
            self.base = base_model
            self.attr = fixed_attr

        def forward(self, x):
            # Replica atributos según tamaño del batch x
            attr_expanded = self.attr.repeat(x.size(0), 1)
            return self.base(x, attr_expanded)

    wrapped_model = Wrapper(model, attr)

    def batch_predict(images):
        images_tensor = torch.stack([transforms.ToTensor()(img).to(DEVICE) for img in images])
        outputs = wrapped_model(images_tensor)
        return torch.softmax(outputs, dim=1).detach().cpu().numpy()

    rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb_img,
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp / 255.0, mask)
    return lime_img, target_class


def visualize_combined_explanations(model, dataset, idx):
    img_tensor, attr_tensor, label = dataset[idx]

    cam_img, pred_class = generate_gradcam(model, img_tensor, attr_tensor)
    lime_img, _ = generate_lime_explanation(model, img_tensor, attr_tensor, pred_class)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(cam_img)
    axs[0].set_title(f"Grad-CAM (Pred: {pred_class})")
    axs[0].axis('off')

    axs[1].imshow(lime_img)
    axs[1].set_title(f"LIME (Pred: {pred_class})")
    axs[1].axis('off')

    plt.show()
