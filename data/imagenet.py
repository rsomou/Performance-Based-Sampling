import os
import requests
import zipfile
from io import BytesIO
from PIL import Image

DATASET_NAME = "tiny-imagenet-200"
IMAGES_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_dataset(base_path):
    """
    Downloads Tiny ImageNet if not already present.
    Expects the dataset to extract to a folder with the structure:
      ./data/tiny-imagenet-200/
          ├─ train/
          └─ val/
    """
    dataset_dir = os.path.join(base_path, DATASET_NAME)
    if os.path.exists(dataset_dir):
        print("Tiny ImageNet dataset already present.")
        return

    os.makedirs(base_path, exist_ok=True)
    print(f"Downloading {IMAGES_URL} ...")
    response = requests.get(IMAGES_URL, stream=True)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(base_path)
    print(f"Downloaded and extracted {DATASET_NAME} to {base_path}.")

def get_dataset_imagenet(base_path):
    """
    Loads Tiny ImageNet into a dictionary with:
      {
        "train": [{"image": PIL.Image, "label": int}, ...],
        "valid": [{"image": PIL.Image, "label": None}, ...]
      }
    Training images are read from the class subdirectories under train/
    and validation images are loaded from val/images/ (without labels).
    NOTE: All images are loaded as PIL objects.
    """
    download_dataset(base_path)
    dataset_dir = os.path.join(base_path, DATASET_NAME)
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # Build a mapping from class names to index based on the train folder names.
    classes = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    dataset = {"train": [], "valid": []}

    # Load training images with labels.
    for cls_name in classes:
        cls_idx = class_to_idx[cls_name]
        images_dir = os.path.join(train_dir, cls_name, "images")
        for fname in os.listdir(images_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(images_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                except Exception as e:
                    print(f"Error loading image {fpath}: {e}")
                    continue
                dataset["train"].append({
                    "image": img,
                    "label": cls_idx
                })

    # Load validation images without labels.
    val_images_dir = os.path.join(val_dir, "images")
    for fname in os.listdir(val_images_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(val_images_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
            except Exception as e:
                print(f"Error loading image {fpath}: {e}")
                continue
            dataset["valid"].append({
                "image": img,
                "label": None
            })

    return dataset

if __name__ == "__main__":
    base_path = "./data"
    dataset = get_dataset_imagenet(base_path)
    print(f"Loaded dataset with {len(dataset['train'])} training images and {len(dataset['valid'])} validation images.")
