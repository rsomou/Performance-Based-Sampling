import argparse
import sys
from data.Dataset import get_dataset, ImageDataset, transform
from utils.models import (
    get_model_architecture, 
    get_model_save_path, 
    get_dataset_root,
    MODEL_DIR
)
import torch
from torch.utils.data import DataLoader
import os
import csv
import numpy as np
from utils.adv_utils import evaluate_cluster_variance


def test_model(dataset, model):
    """
    Test model either using clusters or standard evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()


    correct = 0
    total = 0
    
    # Create validation dataloader directly from dataset structure
    valid_ds = ImageDataset({"valid": dataset["valid"]}, 'valid', transform=transform)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training model parameters
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--dataset_params", nargs='+', type=str, default=[], help="corruption type")

    parser.add_argument("--model", type=str, default="resnet-s")
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--random_w", action="store_true", default=False)

    parser.add_argument("--model_file", type=str, default="", help="model file path to load directly")
    parser.add_argument("--cluster_assignment_file", type=str, default="")



    # Clustering and Sampling Parameters
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--atoms", type=int, default=50)
    parser.add_argument("--sparsity", type=int, default=15)
    parser.add_argument("--base", type=str, default="resnet-s")
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--sampling_formula", type=str, default="exp_dis")
    
    # Variance calculation mode
    parser.add_argument("--eval_var", action="store_true", default=False)

    """
  
    getting model architecture parameters: model, dataset, random_w
    model loading parameters: model_file
    cluster assignment loading for variance calculation: cluster_assignment_file (use eval_var to enter this branch)
    clustering parameters: atoms, entropy, metric, sparsity, epochs (not needed in this file)
    dataset loading parameters: dataset, dataset_params

    """

    args = parser.parse_args()

    if args.model_file == "":
        print("Input Model File")
        sys.exit(1)

    if args.eval_var and args.cluster_assignment_file == "":
        print("If you're using eval_var, provide a assignment file")
        sys.exit(1)

    # Get dataset and model
    dataset = get_dataset(args)
    
    # Get model and load weights
    model = get_model_architecture(args)

    save_path = args.model_file


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)
    
    if(args.eval_var):
        mean_v, class_vs, min_acc, sizes = evaluate_cluster_variance(args.cluster_assignment_file, model, dataset['train'])
        print(f"Mean Variance from {save_path}: {mean_v:.4f}")
        for i,c_v in enumerate(class_vs):
            print(f"Class {i} Variance: {c_v}", end=", " if i<len(class_vs)-1 else "")
            print("\n")

        for i in range(len(min_acc)):
            print(f"Class {i} Minimum Accuracy: {min_acc[i]}", end=", " if i<len(class_vs)-1 else "")
            print("\n")
            print(f"Class {i} Sizes: ")
            print("\n")
            for k,v in sizes[i].items():
                print(f"Cluster {k} Size: {v}")
    else:   
        # Evaluate model performance
        print(f"Evaluating model {save_path} performance")
        test_model(dataset, model)
