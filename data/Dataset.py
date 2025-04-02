from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Lambda
import torchvision.transforms as transforms
from PIL import Image
import random
import torch
from data.celebA import get_dataset_celebA
from data.cifar100 import get_dataset_cifar
from data.cifar100c import get_dataset_cifarc
from data.imagenet import get_dataset_imagenet
from data.tinyimagenetc import get_dataset_tinyimagenet_c

# Keep your existing transforms and classes
transform = transforms.Compose([
    Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
CLIP_transform = transforms.Compose([
    Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def custom_collate_fn(batch):
    labels = []
    images = []
    for item in batch:

        image, lab = CLIP_transform(item['image']), item['label']   
        image = torch.squeeze(image) 
        images.append(image)
        labels.append(lab)

    return torch.stack(images, dim=0), torch.Tensor(labels)

class ImageDataset(Dataset):
    # Keep existing implementation
    def __init__(self, dataset, split, transform):
        self.dataset = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageDatasetSampled(Dataset):
    # Keep existing implementation
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]]['image']
        label = self.dataset[self.indices[idx]]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

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

# Keep other existing functions
def generate_train_dataset(dataset, sampling_ratios, baseline=False):


    assert len(dataset) == len(sampling_ratios), (
        f"Dataset length ({len(dataset)}) must match sampling_ratios length ({len(sampling_ratios)})"
    )
    
    new_dataset_idxs = []
    for i in range(len(sampling_ratios)):
        rand_val = random.random()
        if baseline:
            new_dataset_idxs.append(i)
            continue
        if sampling_ratios[i] < 1:
            if rand_val <= sampling_ratios[i]:
                new_dataset_idxs.append(i)
        elif sampling_ratios[i] == 1.0:
            new_dataset_idxs.append(i)
        else:
            new_dataset_idxs.append(i)
            if rand_val <= (sampling_ratios[i]-1):
                new_dataset_idxs.append(i)
    return ImageDatasetSampled(dataset, new_dataset_idxs, transform=transform)