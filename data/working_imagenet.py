import os
import numpy as np
import requests
import zipfile
import pandas as pd
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Configuration constants
IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
DATASET_NAME = 'tiny-imagenet-200'
NUM_CLASSES = 200

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

def process_batch(image_tensor, label, bbox, split, dataset, bbox_data, count_dict):
    """Process a single batch of data and update the relevant data structures."""
    image_tensor = image_tensor.squeeze(0)
    label = label.item()
    image = transforms.ToPILImage()(image_tensor)
    
    dataset[split].append({
        'image': image,
        'label': label
    })
    bbox_data[split].append(bbox)
    count_dict[split][label] += 1

class TinyImageNetDataset(Dataset):
    def __init__(self, base_path, is_train=True):
        self.base_path = os.path.join(base_path, DATASET_NAME)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Set up paths and load appropriate data
        if is_train:
            self._setup_train_data()
        else:
            self._setup_val_data()

    def _setup_train_data(self):
        """Set up training data and bounding boxes."""
        self.root_dir = os.path.join(self.base_path, 'train')
        self.classes = [d for d in os.listdir(self.root_dir) 
                       if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.classes))}
        
        self.images = []
        self.labels = []
        self.bboxes = []
        bbox_data = {}
        
        for class_name in self.classes:
            # Load bounding box data
            bbox_path = os.path.join(self.root_dir, class_name, f"{class_name}_boxes.txt")
            with open(bbox_path) as f:
                for line in f:
                    img_name, x, y, w, h = line.strip().split()
                    bbox_data[img_name] = [int(x), int(y), int(w), int(h)]
            
            # Load images
            class_dir = os.path.join(self.root_dir, class_name, 'images')
            for image_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, image_name))
                self.labels.append(self.class_to_idx[class_name])
                self.bboxes.append(bbox_data[image_name])

    def _setup_val_data(self):
        """Set up validation data and bounding boxes."""
        self.root_dir = os.path.join(self.base_path, 'val')
        
        # Load validation annotations
        self.val_annotations = pd.read_csv(
            os.path.join(self.root_dir, 'val_annotations.txt'),
            sep='\t',
            header=None,
            names=['File', 'Class', 'X', 'Y', 'W', 'H']
        )
        
        # Get class mapping from training data
        train_path = os.path.join(self.base_path, 'train')
        train_classes = sorted([d for d in os.listdir(train_path) 
                             if os.path.isdir(os.path.join(train_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(train_classes)}
        
        # Set up data lists
        self.images = [os.path.join(self.root_dir, 'images', f) 
                      for f in self.val_annotations['File']]
        self.labels = [self.class_to_idx[cls] for cls in self.val_annotations['Class']]
        self.bboxes = self.val_annotations[['X', 'Y', 'W', 'H']].values.tolist()

    def _load_and_transform_image(self, image_path):
        """Load and transform an image, handling errors."""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_tensor = self._load_and_transform_image(self.images[idx])
        return (image_tensor, 
                torch.tensor(self.labels[idx]), 
                self.bboxes[idx])

def get_dataset_imagenet(base_path):
    """Load and process the Tiny ImageNet dataset."""
    download_dataset(IMAGES_URL, base_path)
    
    # Initialize data structures
    dataset = {'train': [], 'valid': []}
    bbox_data = {'train': [], 'valid': []}
    count_dict = {'train': {}, 'valid': {}}
    
    # Initialize counts
    for i in range(NUM_CLASSES):
        count_dict['train'][i] = 0
        count_dict['valid'][i] = 0
    
    # Process training data
    train_loader = DataLoader(TinyImageNetDataset(base_path, is_train=True), 
                            batch_size=1, shuffle=False)
    for batch in train_loader:
        process_batch(*batch, 'train', dataset, bbox_data, count_dict)
    
    # Process validation data
    val_loader = DataLoader(TinyImageNetDataset(base_path, is_train=False), 
                          batch_size=1, shuffle=False)
    for batch in val_loader:
        process_batch(*batch, 'valid', dataset, bbox_data, count_dict)
    
    return dataset, bbox_data, count_dict