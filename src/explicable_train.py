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


class MultiModalResNet(nn.Module):
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
        return self.classifier(fused)


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
