import os
import requests
import zipfile
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATASET_NAME = 'tiny-imagenet-200'
IMAGE_SIZE = 64

def download_dataset(url, base_path):
    """Downloads and extracts the Tiny ImageNet dataset if not already present."""
    dataset_path = os.path.join(base_path, DATASET_NAME)
    if os.path.exists(dataset_path):
        print('Dataset already downloaded...')
        return
    
    os.makedirs(base_path, exist_ok=True)
    print('Downloading ' + url)
    response = requests.get(url, stream=True)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(base_path)

class TinyImageNetDataset(Dataset):
    def __init__(self, base_path, is_train=True):
        self.base_path = os.path.join(base_path, DATASET_NAME)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.is_train = is_train
        
        if is_train:
            self.root_dir = os.path.join(self.base_path, 'train')
            self._setup_train_data()
        else:
            self.root_dir = os.path.join(self.base_path, 'val')
            self._setup_val_data()

    def _setup_train_data(self):
        """Set up training data."""
        self.classes = sorted([d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name, 'images')
            for image_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, image_name))
                self.labels.append(self.class_to_idx[class_name])

    def _setup_val_data(self):
        """Set up validation data."""
        val_annotations_file = os.path.join(self.root_dir, 'val_annotations.txt')
        self.images = []
        self.labels = []
        
        # Get class mapping from training data
        train_path = os.path.join(self.base_path, 'train')
        train_classes = sorted([d for d in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(train_classes)}
        
        # Read validation annotations
        with open(val_annotations_file, 'r') as f:
            for line in f:
                img_name, class_name = line.strip().split()[:2]
                img_path = os.path.join(self.root_dir, 'images', img_name)
                if os.path.exists(img_path):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def _load_and_transform_image(self, image_path):
        """Load and transform an image."""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self._load_and_transform_image(self.images[idx])
        return image, self.labels[idx]

def get_dataset_imagenet(path):
    """Load and process the Tiny ImageNet dataset."""
    download_dataset(IMAGES_URL, path)
    
    dataset = {"train": [], "valid": []}
    
    train_dataset = TinyImageNetDataset(path, is_train=True)
    val_dataset = TinyImageNetDataset(path, is_train=False)
    
    for img, lab in train_dataset:
        dataset["train"].append({"image": img, "label": lab})
    
    for img, lab in val_dataset:
        dataset["valid"].append({"image": img, "label": lab})
    
    return dataset