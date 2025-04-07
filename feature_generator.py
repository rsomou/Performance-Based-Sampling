import argparse
import torch
from torch.utils.data import DataLoader
from data.Dataset import get_dataset, ImageDataset, CLIP_transform
from transformers import AutoProcessor, CLIPModel
import csv
from tqdm import tqdm
import os
from utils.adv_utils import generate_features

FEATURE_DIR = './features'
def extract_clip_features(dataset, split, output_path, batch_size=32, num_workers=8):

    # Setup CLIP model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", do_rescale=False)
    model.eval()

    # Setup data loader
    ds = ImageDataset(dataset, split, transform=CLIP_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory = True, num_workers = num_workers)

    # Extract and save features
    print(f"Extracting CLIP features for {split} set...")
    with torch.no_grad(), open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for inputs, _ in tqdm(loader, desc="Processing batches"):

            processed_inputs = processor(images=inputs, return_tensors="pt").to(device)
            
            # Get CLIP features
            outputs = model.get_image_features(**processed_inputs)
            
            # Save features
            for features in outputs:
                writer.writerow(features.cpu().tolist())

def main():
    parser = argparse.ArgumentParser(description='Extract feature vectors from dataset')
    parser.add_argument("--feature_type", type=str, default='CLIP')
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    args = parser.parse_args()

    # Get dataset and setup output path
    dataset = get_dataset(args)

    os.makedirs(FEATURE_DIR, exist_ok='true')

    if(args.feature_type == 'CLIP'):
        output_path = f"{FEATURE_DIR}/CLIP_{args.dataset}_{args.split}.csv"

        # Extract features
        extract_clip_features(
            dataset, 
            args.split, 
            output_path,
            args.batch_size, 
            args.num_workers
        )

    elif (args.feature_type == 'LL'):
        output_path = f"{FEATURE_DIR}/LL_{args.dataset}_{args.split}.csv"
        generate_features(
            dataset,
            args.split,
            output_path
        )


if __name__ == "__main__":
    main()