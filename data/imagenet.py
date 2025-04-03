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
    Returns a dictionary:
        {
          "train": [
             {"image": PIL.Image, "label": int},
             {"image": PIL.Image, "label": int},
             ...
          ],
          "valid": [
             {"image": PIL.Image, "label": int},
             ...
          ]
        }
    """
    download_dataset(base_path)
    dataset_dir = os.path.join(base_path, DATASET_NAME)
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # 1. Build class -> idx mapping from train folder
    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Prepare our final structure
    dataset = {"train": [], "valid": []}

    # 2. Populate "train" list with {"image": <PIL>, "label": <int>}
    for cls_name in classes:
        cls_idx = class_to_idx[cls_name]
        images_dir = os.path.join(train_dir, cls_name, "images")
        for fname in os.listdir(images_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(images_dir, fname)
                # Load the image as PIL
                img = Image.open(fpath).convert("RGB")
                dataset["train"].append({
                    "image": img,
                    "label": cls_idx
                })

    # 3. Populate "valid" list using val_annotations.txt
    val_anno_path = os.path.join(val_dir, "val_annotations.txt")
    val_df = pd.read_csv(
        val_anno_path, sep='\t', header=None,
        names=["File", "Class", "X", "Y", "W", "H"]
    )
    for _, row in val_df.iterrows():
        fname = row["File"]
        cls_name = row["Class"]
        if cls_name not in class_to_idx:
            # Ideally should not happen, but just in case
            continue

        fpath = os.path.join(val_dir, "images", fname)
        img = Image.open(fpath).convert("RGB")
        dataset["valid"].append({
            "image": img,
            "label": class_to_idx[cls_name]
        })

    return dataset

if __name__ == "__main__":
    base_path = "./data"
    dataset = get_dataset_imagenet(base_path)
    print(f"Loaded dataset with {len(dataset['train'])} training images and {len(dataset['valid'])} validation images.")
