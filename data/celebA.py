from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os
from collections import defaultdict
from torchvision import datasets

from torchvision import datasets
from collections import defaultdict
from tqdm import tqdm
import gc


def get_dataset_celebA(path, samples_per_class = 27000):
    """
    Get CelebA dataset using torchvision
    Returns formatted dataset with images and blond hair labels (attr_index 9 for blonde hair)
    Includes all three splits: train, valid, test
    """
    print("Loading CelebA dataset...")
    
    def process_split(split):
        print(f"\nProcessing {split} split...")
        dataset = datasets.CelebA(root=path, split=split, target_type='attr', download=True)
        images, labels = [], []
        label_counts = {0:0,1:0}
        
        for img, target in tqdm(dataset):
            with img: # This will auto-close the image
                label = int(target[9])
                if(label_counts[0]>samples_per_class and label_counts[1]>samples_per_class):
                    break
                images.append(img.copy()) # Make a copy before closing
                labels.append(label)
                label_counts[label] += 1
            
        return images, labels, dict(label_counts)

    # Process all splits
    splits = ['train', 'valid', 'test']
    all_data = {}
    all_counts = {}
    
    for split in splits:
        images, labels, counts = process_split(split)
        all_data[split] = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
        all_counts[split] = counts
    
    return all_data

if __name__ == "__main__":
    try:
        gc.collect()
        dataset, _, counts = get_dataset_celebA("./celebA")
        print("\nDataset loaded successfully!")
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['valid'])}")
        print("\nLabel distribution:")
        print("Training set:", counts['train'])
        print("Validation set:", counts['valid'])
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")