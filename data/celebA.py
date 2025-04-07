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


def get_dataset_celebA(path, samples_per_class=24000):
    """
    Lazily load the CelebA dataset using torchvision.
    
    Instead of loading all images into memory, this function reads the list
    of image file names and their corresponding attributes from the downloaded
    dataset. It then filters the data based on the desired number of samples per class 
    for blond hair (attribute index 9). The returned dataset for each split 
    contains dictionaries with:
    
        {"image": <file_path>, "label": <blond hair label>}
    
    Parameters:
        path (str): Root directory where CelebA is (or will be) downloaded.
        samples_per_class (int): Maximum number of samples to keep per class.
        
    Returns:
        all_data (dict): A dictionary with splits ('train', 'valid', 'test') as keys.
        all_counts (dict): A dictionary with the final label counts for each split.
    """
    print("Loading CelebA dataset lazily...")

    splits = ['train', 'valid', 'test']
    all_data = {}
    all_counts = {}

    for split in splits:
        print(f"\nProcessing {split} split lazily...")
        # Create a CelebA dataset for the given split.
        split_dataset = datasets.CelebA(root=path, split=split, target_type="attr", download=True)
        # 'filename' is a list of image file names.
        filenames = split_dataset.filename
        # 'attr' is a list (or array) of attribute vectors for each image.
        # We extract the blond hair attribute (index 9) and convert to int.
        labels = [int(attr[9]) for attr in split_dataset.attr]
        
        # We'll collect file paths and labels until we reach samples_per_class per label.
        label_counts = {0: 0, 1: 0}
        data = []
        for fname, label in tqdm(zip(filenames, labels), total=len(filenames), desc=f"Filtering {split}"):
            if label_counts[label] >= samples_per_class:
                continue
            # Build the full file path. The CelebA dataset stores images in the base_folder.
            img_path = os.path.join(split_dataset.base_folder, fname)
            data.append({"image": img_path, "label": label})
            label_counts[label] += 1

        all_data[split] = data
        all_counts[split] = label_counts

    return all_data, all_counts
