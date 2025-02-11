import os
import resource
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import defaultdict
from tqdm import tqdm
import argparse
import gc

# Increase system file limit
def increase_file_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, data, split, transform=None):
        self.data = data[split]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def test_model(dataset, model, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    # Create validation dataloader with smaller batch size
    valid_ds = ImageDataset({split: dataset}, split, transform=transform)
    valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=2)
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Clear some memory
            del images, labels, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return correct, total

def get_test_celebA(path):
    print("Loading CelebA dataset...")
    
    # Process data in chunks to avoid opening too many files
    chunk_size = 1000
    test_data_attr_30 = defaultdict(list)
    
    # Create dataset
    dataset = datasets.CelebA(root=path, split='test', target_type='attr', download=True)
    total_samples = len(dataset)
    
    for i in tqdm(range(0, total_samples, chunk_size)):
        chunk_end = min(i + chunk_size, total_samples)
        
        for idx in range(i, chunk_end):
            try:
                image, labels = dataset[idx]
                test_data_attr_30[labels[30].item()].append({
                    "image": transforms.functional.pil_to_tensor(image),  # Convert to tensor immediately
                    "label": labels[9].item()
                })
                
                # Explicitly close the image
                image.close()
                
            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                continue
            
        # Force garbage collection after each chunk
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert defaultdict to list
    return [test_data_attr_30[0], test_data_attr_30[1]]

def main():
    # Increase file limit at the start
    try:
        increase_file_limit()
    except Exception as e:
        print(f"Warning: Could not increase file limit: {str(e)}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet-s")
    parser.add_argument("--dataset", type=str, default="celebA")
    parser.add_argument("--random_w", action="store_true", default=False)
    args = parser.parse_args()
    
    # Initialize model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if not args.random_w else None)
    model.fc = nn.Linear(
        in_features=2048,
        out_features=2,
        bias=True
    )
    
    # Load model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model_file, map_location=device)
    model.load_state_dict(state)
    
    try:
        # Get test data and evaluate
        test_data = get_test_celebA("./data/celebA")
        correct_0, total_0 = test_model(test_data[0], model, "test")
        correct_1, total_1 = test_model(test_data[1], model, "test")
        
        print("\nConfusion Matrix:")
        print(f"{correct_0:>6d} {total_0-correct_0:>6d}")
        print(f"{total_1-correct_1:>6d} {correct_1:>6d}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    finally:
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()