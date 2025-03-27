# models.py

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

# Constants
IMAGENET_ROOT = "./data/tiny-imagenet-200"
CIFAR_ROOT = "./data/cifar100"
CELEBA_ROOT = "./data/celebA"
CIFARC_ROOT = "./data/cifar100c"
TINYIMAGENETC_ROOT = "./data/tinyimagenetc"
MODEL_DIR = "./models"

def get_model_architecture(args):
    """
    Creates model architecture based on model name and dataset.
    
    Args:
        model_name (str): Name of the model ('deit' or 'resnet')
        dataset_name (str): Name of the dataset ('imagenet', 'cifar100', or 'celebA')
    
    Returns:
        torch.nn.Module: Initialized model
    """

    model_name = args.model
    dataset_name = args.dataset
    random = args.random_w

    if model_name == "deit":
        model = vit_b_16(weights=ViT_B_16_Weights) if not random else vit_b_16(weights=None)
        output_features = {
            "imagenet": 200,
            "cifar100": 100,
            "celebA": 2
        }
        if dataset_name not in output_features:
            raise ValueError(f"Invalid dataset: {dataset_name}")
            
        model.heads.head = nn.Linear(
            in_features=768, 
            out_features=output_features[dataset_name], 
            bias=True
        )
    else:  # ResNet
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if not random else resnet50(weights=None)
        output_features = {
            "imagenet": 200,
            "cifar100": 100,
            "celebA": 2
        }
        if dataset_name not in output_features:
            raise ValueError(f"Invalid dataset: {dataset_name}")
            
        model.fc = nn.Linear(
            in_features=2048, 
            out_features=output_features[dataset_name], 
            bias=True
        )
    
    return model

def get_dataset_root(dataset_name):
    """
    Gets the root directory for a given dataset.
    
    Args:
        dataset_name (str): Name of the dataset
    
    Returns:
        str: Path to dataset root
    """
    roots = {
        "imagenet": IMAGENET_ROOT,
        "cifar100": CIFAR_ROOT,
        "celebA": CELEBA_ROOT,
        "cifar100c": CIFARC_ROOT,
        "tinyimagenetc": TINYIMAGENETC_ROOT
    }
    if dataset_name not in roots:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    return roots[dataset_name]

def get_model_save_path(args):
    """
    Generates the model save/load path based on arguments.
    
    Args:
        args: ArgumentParser arguments
        feat_name (str): Name of features
    
    Returns:
        str: Path where model should be saved/loaded
    """
    if args.baseline == False:
        return f"{MODEL_DIR}/finetuned-{args.dataset}-{args.model}-{args.epochs}-{args.sampling_formula}-{args.entropy}-{args.atoms}-{args.sparsity}.pth"
    else:
        return f"{MODEL_DIR}/baseline-{args.dataset}-{args.model}-{args.epochs}.pth"

def load_saved_model(model, path):
    """
    Loads saved model weights.
    
    Args:
        model (torch.nn.Module): Model architecture
        path (str): Path to saved weights
        
    Returns:
        torch.nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(path))
    return model
