from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
import torch
from data.celebA import get_dataset_celebA
from data.cifar100 import get_dataset_cifar
from data.cifar100c import get_dataset_cifarc
from data.imagenet import get_dataset_imagenet
from data.tinyimagenetc import get_dataset_tinyimagenet_c

transform = transforms.Compose([
    Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
CLIP_transform = transforms.Compose([
    Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def clip_collate_fn(batch):
    labels = []
    images = []
    for item in batch:
        # Use the unified access: load if file path exists, otherwise use preloaded image
        if 'image_path' in item:
            image = Image.open(item['image_path']).convert("RGB")
        else:
            image = item['image']
        image = CLIP_transform(image)
        image = torch.squeeze(image)
        images.append(image)
        labels.append(item['label'])
    return torch.stack(images, dim=0), torch.Tensor(labels)

def custom_collate_fn(batch):
    labels = []
    images = []
    for item in batch:
        # Unified handling for both file paths and preloaded images
        if 'image_path' in item:
            image = Image.open(item['image_path']).convert("RGB")
        else:
            image = item['image']
        image = transform(image)
        image = torch.squeeze(image)
        images.append(image)
        labels.append(item['label'])
    return torch.stack(images, dim=0), torch.Tensor(labels)

class ImageDataset(Dataset):
    """
    Unified Dataset class that works with both lazy-loaded file paths and preloaded images.
    """
    def __init__(self, dataset, split, transform):
        # dataset is expected to be a dictionary with splits as keys.
        self.dataset = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        # Check if the entry is lazy (using a file path) or already loaded
        if 'image_path' in entry:
            image = Image.open(entry['image_path']).convert("RGB")
        elif 'image' in entry:
            image = entry['image']
        else:
            raise ValueError("Dataset entry must have either 'image_path' or 'image' key.")
        if self.transform:
            image = self.transform(image)
        return image, entry['label']

class ImageDatasetSampled(Dataset):
    """
    A dataset for a sampled subset.
    """
    def __init__(self, dataset, indices, transform=None):
        # dataset should be a list-like structure with entries containing image info.
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        entry = self.dataset[actual_idx]
        if 'image_path' in entry:
            image = Image.open(entry['image_path']).convert("RGB")
        elif 'image' in entry:
            image = entry['image']
        else:
            raise ValueError("Dataset entry must have either 'image_path' or 'image' key.")
        if self.transform:
            image = self.transform(image)
        return image, entry['label']
    
# Dataset paths
DATASET_PATHS = {
    "imagenet": "./data",
    "cifar100": "./data/cifar100",
    "celebA": "./data/celebA",
    "tinyimagenetc": "./data/tinyimagenetc",
    "cifar100c": "./data/cifar100c"
}

def get_dataset(args):
    """
    Centralized dataset loading function
    """
    dataset_name = args.dataset
    
    if dataset_name == "imagenet":
        return get_dataset_imagenet(DATASET_PATHS["imagenet"])
    elif dataset_name == "cifar100":
        return get_dataset_cifar(DATASET_PATHS["cifar100"])
    elif dataset_name == "cifar100c":
        return get_dataset_cifarc(DATASET_PATHS["cifar100c"], args.ctype)
    elif dataset_name == "celebA":
        return get_dataset_celebA(DATASET_PATHS["celebA"])
    elif dataset_name == "tinyimagenetc":
        return get_dataset_tinyimagenet_c(DATASET_PATHS["tinyimagenetc"], args.ctype, args.cdeg)
    else:
        raise ValueError("Invalid Dataset")

# Vectorized Sample Ratio Generation

def generate_train_dataset(dataset, sampling_ratios, baseline=False):
    sampling_ratios = np.array(sampling_ratios)
    n = len(sampling_ratios)
    indices = np.arange(n)
    
    if baseline:
        new_dataset_idxs = indices.tolist()
    else:
        # Generate a random value for each index in one go
        rand_vals = np.random.rand(n)
        # Include an index once if:
        #   - The ratio is >= 1 (always include), or
        #   - The ratio is < 1 and the random value is less than or equal to the ratio.
        include_once = (sampling_ratios >= 1) | ((sampling_ratios < 1) & (rand_vals <= sampling_ratios))
        # For ratios > 1, add a duplicate if the random value is less than or equal to (ratio - 1)
        add_duplicate = (sampling_ratios > 1) & (rand_vals <= (sampling_ratios - 1))
        # Build the index list: first the ones to include once, then add duplicates.
        new_dataset_idxs = indices[include_once].tolist() + indices[add_duplicate].tolist()

    return ImageDatasetSampled(dataset, new_dataset_idxs, transform=transform)