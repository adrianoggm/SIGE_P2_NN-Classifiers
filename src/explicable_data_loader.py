import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from config import DATA_DIR_X20, DATA_DIR_X200, BATCH_SIZE, MAIN_DATASET


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
    Configura rutas y transformaciones según MAIN_DATASET:
      - 'x20': principal en DATA_DIR_X20
      - 'x200': principal en DATA_DIR_X200

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


class CUBMultimodalDataset(Dataset):
    """
    Dataset que carga imagen, atributos y etiqueta.
    """
    def __init__(self, root_dir, attr_txt, transform=None):
        self.img_folder = datasets.ImageFolder(root_dir, transform=transform)
        self.attr_dict = self._parse_attr_file(attr_txt)
        all_attrs = set(a for attrs in self.attr_dict.values() for a in attrs)
        self.attr2idx = {attr: i for i, attr in enumerate(sorted(all_attrs))}
        self.attr_dim = len(self.attr2idx)

    def _parse_attr_file(self, path):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                rel_path, rest = line.strip().split(':', 1)
                entries = [e.strip() for e in rest.split(',') if e.strip()]
                attrs = [ent.replace(':','_').replace(' ', '_') for ent in entries]
                d[rel_path] = attrs
        return d

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, idx):
        path, label = self.img_folder.samples[idx]
        image = self.img_folder.loader(path)
        if self.img_folder.transform:
            image = self.img_folder.transform(image)
        rel_path = os.path.relpath(path, self.img_folder.root)
        attr_vec = torch.zeros(self.attr_dim, dtype=torch.float)
        for attr in self.attr_dict.get(rel_path, []):
            if attr in self.attr2idx:
                attr_vec[self.attr2idx[attr]] = 1.0
        return image, attr_vec, label


def load_datasets(attr_txt):
    """
    1) Carga dataset multimodal
    2) Divide en train/val 80/20
    3) Añade augment solo a train
    """
    main_dir, _, std_tf, aug_tf = get_dataset_configs()
    full = CUBMultimodalDataset(main_dir, attr_txt, transform=std_tf)
    n = len(full)
    train_n = int(0.8 * n)
    val_n = n - train_n
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full, [train_n, val_n], generator=g)
    aug = CUBMultimodalDataset(main_dir, attr_txt, transform=aug_tf)
    train_aug_ds = Subset(aug, train_ds.indices)
    train_full = ConcatDataset([train_ds, train_aug_ds])
    return full, train_ds, val_ds, train_aug_ds, train_full


def custom_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    attrs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, attrs, labels


def get_dataloaders(attr_txt):
    """
    Retorna DataLoaders para entrenamiento y validación.
    """
    _, train_ds, val_ds, _, train_full = load_datasets(attr_txt)
    train_loader = DataLoader(
        train_full, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate
    )
    return train_loader, val_loader
