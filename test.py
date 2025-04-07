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
    valid_ds = ImageDataset({split: dataset["valid"]}, 'valid', transform=transform)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=True, pin_memory = True, num_workers = 4, collate_fn = custom_collate_fn)

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

    parser.add_argument("--split", type=str, default="valid", help="Use Test Split if available")

    parser.add_argument("--model_file", type=str, default="", help="model file path to load directly")
    parser.add_argument("--cluster_assignment_file", type=str, default="")



    # Clustering and Sampling Parameters
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--atoms", type=int, default=50)
    parser.add_argument("--sparsity", type=int, default=15)
    parser.add_argument("--base", type=str, default="resnet-s")
    parser.add_argument("--entropy", type=float, default=0)
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
        output_file = f"{get_model_save_path(args)[:-4]}-variance-output.txt"
        with open(output_file, "w") as file:
            mean_v, class_vs, min_acc, sizes = evaluate_cluster_variance(
                args.cluster_assignment_file, model, dataset['train']
            )
            file.write(f"Mean Variance from {save_path}: {mean_v:.4f}\n")
            
            for i, c_v in enumerate(class_vs):
                file.write(f"Class {i} Variance: {c_v}")
                if i < len(class_vs) - 1:
                    file.write(', ')
            file.write("\n\n")
            
            for i in range(len(min_acc)):
                file.write(f"Class {i} Minimum Accuracy: {min_acc[i]}")
                if i < len(min_acc) - 1:
                    file.write(", ")
                file.write("\n")
                file.write(f"Class {i} Statistics:\n")
                for k, v in sizes[i].items():
                    file.write(f"Cluster {k} Stats: Size {v[0]}, Acc: {v[1]} \n")
                file.write("\n")
        print(f"Variance evaluation results saved to {output_file}")
    else:   
        # Evaluate model performance
        print(f"Evaluating model {save_path} performance")
        test_model(dataset, model)
