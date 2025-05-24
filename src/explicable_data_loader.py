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
    - aug_transforms: flip, rotation ±15°, affine zoom (0.8–1.2)
      each separate, generating 3 augmented images per sample
    """
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # Define each augmentation separately
    flip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ])
    rotate_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    affine_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomAffine(
            degrees=0,
            scale=(0.8, 1.2),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
    ])

    aug_transforms = [flip_transform, rotate_transform, affine_transform]
    return standard_transform, aug_transforms


def get_dataset_configs(main_dataset_choice=None):
    """
    Select main directory based on MAIN_DATASET:
      - 'x20': DATA_DIR_X20, secondary DATA_DIR_X200
      - 'x200': DATA_DIR_X200, secondary DATA_DIR_X20
    Returns main_data_dir, secondary_data_dir, standard_transform, aug_transforms
    """
    if main_dataset_choice is None:
        main_dataset_choice = MAIN_DATASET

    standard_transform, aug_transforms = get_transformations()

    if main_dataset_choice == "x20":
        main_data_dir = DATA_DIR_X20
        secondary_data_dir = DATA_DIR_X200
    elif main_dataset_choice == "x200":
        main_data_dir = DATA_DIR_X200
        secondary_data_dir = DATA_DIR_X20
    else:
        raise ValueError("MAIN_DATASET debe ser 'x20' o 'x200'.")

    return main_data_dir, secondary_data_dir, standard_transform, aug_transforms


def load_attributes_txt(path):
    """
    Load attribute names from attributes.txt: each line '<id> <name>'.
    Returns list of normalized attribute names.
    """
    attrs = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                attrs.append(parts[1].replace(' ', '_').replace(':', '_'))
    return attrs


def load_image_id_map(images_txt):
    """
    Load images.txt mapping: '<img_id> <rel_path>'.
    Returns dict rel_path -> img_id.
    """
    d = {}
    with open(images_txt, 'r') as f:
        for line in f:
            img_id, rel_path = line.strip().split()
            d[os.path.basename(rel_path)] = int(img_id)
    return d


def load_image_attributes(image_attr_path, num_attrs):
    """
    Load image_attribute_labels.txt: '<img_id> <attr_id> <is_localized> <is_present> <confidence>'.
    Returns dict img_id -> binary tensor of attributes.
    """
    import collections
    attr_data = collections.defaultdict(lambda: torch.zeros(num_attrs, dtype=torch.float))
    with open(image_attr_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            attr_id = int(parts[1]) - 1
            is_present = int(parts[3])
            if is_present:
                attr_data[img_id][attr_id] = 1.0
    return dict(attr_data)


class CUBMultimodalDataset(Dataset):
    """
    Multimodal CUB: image + attribute vector + label.
    """
    def __init__(self, root_dir, transform=None):
        self.img_folder = datasets.ImageFolder(root_dir, transform=transform)
        self.imgs = self.img_folder.samples

        # Load attributes
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

        rel_name = os.path.basename(path)
        img_id = self.image_id_map.get(rel_name, None)
        attr_vec = self.image_attrs.get(img_id, torch.zeros(self.attr_dim, dtype=torch.float))
        return image, attr_vec, label


def load_datasets():
    """
    1) Load multimodal dataset with standard transform
    2) Split train/val 80/20 (seed=42)
    3) For each augment transform, create augmented subset of train
    4) Concatenate: train_clean + each augmented subset
    """
    main_dir, _, std_tf, aug_tfs = get_dataset_configs()
    full = CUBMultimodalDataset(main_dir, transform=std_tf)

    # Split
    n = len(full)
    train_n = int(0.8 * n)
    val_n = n - train_n
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full, [train_n, val_n], generator=g)

    # Augmented subsets
    aug_subsets = []
    for tfm in aug_tfs:
        aug_ds = CUBMultimodalDataset(main_dir, transform=tfm)
        aug_subsets.append(Subset(aug_ds, train_ds.indices))

    # Combined train: clean + all aug
    train_full = ConcatDataset([train_ds, *aug_subsets])
    return full, train_ds, val_ds, aug_subsets, train_full


def custom_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    attrs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, attrs, labels


def get_dataloaders():
    _, train_ds, val_ds, _aug_sets, train_full = load_datasets()
    train_loader = DataLoader(train_full, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=custom_collate)
    return train_loader, val_loader