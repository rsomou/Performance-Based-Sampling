from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os
from collections import defaultdict
import torch
from torchvision import datasets
import gc
from data.Dataset import get_dataset, ImageDataset, transform
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import argparse

def test_model(dataset, model, split):
    """
    Test model either using clusters or standard evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    correct = 0
    total = 0
    
    # Create validation dataloader directly from dataset structure
    valid_ds = ImageDataset({split: dataset}, split, transform=transform)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

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

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return correct, total

def get_test_celebA(path):

    print("Loading CelebA dataset...")
    dataset = datasets.CelebA(root=path, split='valid', target_type='attr', download=True)
    
    # Initialize lists using defaultdict
    test_data_attr_30 = defaultdict(list)
    
    for image, labels in tqdm(dataset):
        with image:
            test_data_attr_30[int(labels[30])].append({"image": image.copy(), "label": int(labels[9])})
    
    # Convert defaultdict to list
    return [test_data_attr_30[0], test_data_attr_30[1]]

def bin_celebA(path):

    print("Loading CelebA dataset...")
    dataset = datasets.CelebA(root=path, split='valid', target_type='attr', download=True)
    
    # Initialize lists using defaultdict
    test_data = defaultdict(list)
    
    for image, labels in tqdm(dataset):
        with image:
            test_data[(int(labels[30]),int(labels[9]))].append({"image": image.copy(), "label": int(labels[9])})
    
    # Convert defaultdict to list
    return test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--model",type=str,default="resnet-s")
    parser.add_argument("--dataset",type=str,default="celebA")
    parser.add_argument("--random_w", action="store_true", default=False)

    args = parser.parse_args()


    model= resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if not args.random_w else resnet50(weights=None)            
    model.fc = nn.Linear(
        in_features=2048, 
        out_features=2, 
        bias=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.model_file, map_location=device)
    model.load_state_dict(state)
    
    #test_data = get_test_celebA("./data/celebA")
    test_data = bin_celebA("./data/celebA")

    #correct_0, total_0 = test_model(test_data[0], model, "test")
    #correct_1, total_1 = test_model(test_data[1], model, "test")

    #print(f"Acc when attribute 30 is 0: {correct_0/total_0:.4f} , Err rate when attribute 30 is 0: {(total_0-correct_0)/total_0:.4f} \n")
    #print(f"Acc when attribute 30 is 1: {correct_1/total_1:.4f} , Err rate when attribute 30 is 0: {(total_1-correct_1)/total_1:.4f} \n")

    correct_11, total_11 = test_model(test_data[(1,1)],model, "test")
    correct_10, total_10 = test_model(test_data[(1,0)],model, "test")
    correct_01, total_01 = test_model(test_data[(0,1)],model, "test")
    correct_00, total_00 = test_model(test_data[(0,0)],model, "test")    

    print("Confusion Matrix")
    print(f"dark female (label 9: 0, label 3:0 0) Acc: {correct_00/total_00:.4f} , dark male (label 9: 0, label 30: 1): {correct_01/total_01:.4f} \n")
    print(f"blonde female (label 9: 1, label 30: 0): {correct_10/total_10:.4f} , blond male (label 9: 1, label 30: 1): {correct_11/total_11:.4f} \n")
    

if __name__ == "__main__":
    main()