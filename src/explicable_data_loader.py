import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from config import (
    DATA_DIR_X20,
    DATA_DIR_X200,
    BATCH_SIZE,
    MAIN_DATASET,
    ATTRIBUTES_PATH,
    IMAGE_ATTR_LABELS_PATH,
    IMAGES_TXT_PATH,
)


def get_transformations():
    """
    - standard_transform: resize 224×224 + ToTensor
    - aug_transform: random mirror, rotación ±15° o escala zoom (0.8–1.2) + ToTensor
    """
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    aug_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(
                degrees=0,
                scale=(0.8, 1.2),
                interpolation=InterpolationMode.BILINEAR
            ),
        ]),
        transforms.ToTensor(),
    ])

    return standard_transform, aug_transform


def get_dataset_configs(main_dataset_choice=None):
    """
    Elige la carpeta principal de imágenes según MAIN_DATASET:
      - 'x20': DATA_DIR_X20 contiene ~20 imágenes por clase.
      - 'x200': DATA_DIR_X200 contiene ~200 imágenes por clase.

    Retorna:
      main_data_dir, secondary_data_dir, standard_transform, aug_transform
    """
    if main_dataset_choice is None:
        main_dataset_choice = MAIN_DATASET

    standard_transform, aug_transform = get_transformations()

    if main_dataset_choice == "x20":
        main_data_dir = DATA_DIR_X20
        secondary_data_dir = DATA_DIR_X200
    elif main_dataset_choice == "x200":
        main_data_dir = DATA_DIR_X200
        secondary_data_dir = DATA_DIR_X20
    else:
        raise ValueError("MAIN_DATASET debe ser 'x20' o 'x200'.")

    return main_data_dir, secondary_data_dir, standard_transform, aug_transform


def load_attributes_txt(path):
    """
    Carga la lista de atributos desde attributes.txt.
    Cada línea: "<id> <attr_name>"
    Devuelve lista de nombres de atributos normalizados.
    """
    attrs = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                # Reemplaza espacios y ':' por '_' para consistencia
                attrs.append(parts[1].replace(' ', '_').replace(':', '_'))
    return attrs


def load_image_id_map(images_txt):
    """
    Carga images.txt de CUB:
    Cada línea: "<img_id> <rel_path>"
    Devuelve dict: {rel_path: img_id}
    """
    d = {}
    with open(images_txt, 'r') as f:
        for line in f:
            img_id, rel_path = line.strip().split()
            # Mantiene sólo el nombre de archivo
            d[rel_path] = int(img_id)
    return d


def load_image_attributes(image_attr_path, num_attrs):
    """
    Carga image_attribute_labels.txt de CUB:
    Cada línea: "<img_id> <attr_id> <is_localized> <is_present> <confidence>"
    Devuelve dict: {img_id: tensor binario de atributos}
    """
    import collections
    attr_data = collections.defaultdict(lambda: torch.zeros(num_attrs, dtype=torch.float))
    with open(image_attr_path, 'r') as f:
        for line in f:
            img_id, attr_id, *_rest = line.strip().split()
            img_id = int(img_id)
            attr_id = int(attr_id) - 1  # convertir a índice 0-based
            is_present = int(line.strip().split()[3])
            if is_present:
                attr_data[img_id][attr_id] = 1.0
    return dict(attr_data)


class CUBMultimodalDataset(Dataset):
    """
    Dataset multimodal CUB: imagen + vector binario de 312 atributos + etiqueta.
    """
    def __init__(self, root_dir, transform=None):
        self.img_folder = datasets.ImageFolder(root_dir, transform=transform)
        # Para compatibilidad con visualización
        self.imgs = self.img_folder.samples

        # Cargar mapeos y atributos
        self.attr_names = load_attributes_txt(ATTRIBUTES_PATH)
        self.attr_dim = len(self.attr_names)
        self.image_id_map = load_image_id_map(IMAGES_TXT_PATH)
        self.image_attrs = load_image_attributes(IMAGE_ATTR_LABELS_PATH, num_attrs=self.attr_dim)

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, idx):
        path, label = self.img_folder.samples[idx]
        image = self.img_folder.loader(path)
        if self.img_folder.transform:
            image = self.img_folder.transform(image)

        # Obtener ruta relativa respecto al root
        rel_path = os.path.relpath(path, self.img_folder.root).replace('\\', '/')
        # Extraer sólo el nombre de archivo
        rel_path = os.path.basename(rel_path)

        img_id = self.image_id_map.get(rel_path, None)
        attr_vec = self.image_attrs.get(img_id, torch.zeros(self.attr_dim, dtype=torch.float))
        return image, attr_vec, label


def load_datasets():
    """
    1) Carga dataset multimodal con transformación estándar.
    2) Divide en train/val 80/20 (seed fija 42).
    3) Aplica augment sólo en train y concatena.
    """
    main_dir, _, std_tf, aug_tf = get_dataset_configs()
    full = CUBMultimodalDataset(main_dir, transform=std_tf)
    n = len(full)
    train_n = int(0.8 * n)
    val_n = n - train_n
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full, [train_n, val_n], generator=g)
    aug = CUBMultimodalDataset(main_dir, transform=aug_tf)
    train_aug_ds = Subset(aug, train_ds.indices)
    train_full = ConcatDataset([train_ds, train_aug_ds])
    return full, train_ds, val_ds, train_aug_ds, train_full


def custom_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    attrs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, attrs, labels


def get_dataloaders():
    """
    Retorna DataLoaders para entrenamiento y validación.
    """
    _, train_ds, val_ds, _, train_full = load_datasets()
    train_loader = DataLoader(train_full, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    return train_loader, val_loader
