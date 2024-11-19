import argparse
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
    parser.add_argument("--model", type=str, default="resnet-s")
    parser.add_argument("--model_file", type=str, default="", help="model file path to load directly")
    parser.add_argument("--cluster_assignment_file", type=str, default="")
    parser.add_argument("--epochs", type=int, default=7)

    # Clustering and Sampling Parameters
    parser.add_argument("--atoms", type=int, default=50)
    parser.add_argument("--sparsity", type=int, default=15)
    parser.add_argument("--baseline", action="store_true", default=False) 
    parser.add_argument("--base", type=str, default="resnet-s")
    parser.add_argument("--epsilon", type=float, default=0)
    parser.add_argument("--sampling_formula", type=str, default="exp_dis")
    
    # Variance calculation mode
    parser.add_argument("--eval_var", action="store_true", default=False)

    args = parser.parse_args()

    # Get dataset and model
    dataset, _, counts = get_dataset(args)
    
    # Get model and load weights
    model = get_model_architecture(args.model, args.dataset_t)

    save_path = ""
    if args.model_file == "":
        save_path = get_model_save_path(args, " ".join(args.features_t))
    else:
        save_path = args.model_file

    model.load_state_dict(torch.load(save_path))
    
    if(args.eval_var):
        mean_v,_ = evaluate_cluster_variance(args.cluster_assignment_file, model, dataset['train'])
        print(f"Mean Variance from {save_path}: {mean_v:.4f}")
    else:   
        # Evaluate model performance
        test_model(dataset, model)
