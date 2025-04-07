import argparse
import os 
import random
from torch.utils.data import DataLoader
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from livelossplot import PlotLosses
import sys

from data.Dataset import get_dataset, ImageDataset, ImageDatasetSampled, transform, custom_collate_fn, generate_train_dataset
from utils.models import (
    get_model_architecture, 
    get_model_save_path, 
    get_dataset_root, 
    MODEL_DIR
)
from utils.adv_utils import generate_sampling_ratios


def get_initial_loaders(dataset, sampling_ratios):

    dataloaders = {}
    dataset_sizes = {}

    if 'train' not in dataset:
        raise ValueError("Dataset must contain a 'train' split")

    train_ds = generate_train_dataset(dataset['train'], sampling_ratios)
    dataloaders['train'] = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory = True)
    dataset_sizes['train'] = len(train_ds)

    if 'valid' in dataset:
        valid_ds = ImageDataset(dataset, 'valid', transform=transform)
        dataloaders['valid'] = DataLoader(valid_ds, batch_size=32, shuffle=True, pin_memory = True)
        dataset_sizes['valid'] = len(valid_ds)

    if 'test' in dataset:
        print("No validation set found, using test set for validation")
        valid_ds = ImageDataset(dataset, 'test', transform=transform)
        dataloaders['valid'] = DataLoader(valid_ds, batch_size=32, shuffle=True, pin_memory = True)
        dataset_sizes['valid'] = len(valid_ds)
    
    
    return dataloaders, dataset_sizes

def train_model(device, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(f"\rIteration: {i+1}/{len(dataloaders[phase])}, Loss: {loss.item() * inputs.size(0)}", end="")
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                scheduler.step()  # Step scheduler once per epoch
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        liveloss.update({
            'log loss': avg_loss,
            'val_log loss': val_loss,
            'accuracy': t_acc,
            'val_accuracy': val_acc
        })

        print(f'\nTrain Loss: {avg_loss:.4f} Acc: {t_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Best Val Accuracy: {best_acc}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100") 
    parser.add_argument("--dataset_params", nargs='+', type=str, default=[], help="corruption type")

    parser.add_argument("--model", type=str, default="resnet-s")
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--random_w", action="store_true", default=False)

    parser.add_argument("--cluster_assignment_file", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)

    # Clustering and Sampling Parameters
    parser.add_argument("--atoms", type=int, default=50)
    parser.add_argument("--sparsity", type=int, default=15)
    parser.add_argument("--entropy", type=float, default=0)
    parser.add_argument("--sampling_formula", type=str, default="exp_dis")
    
    """
  
    getting model architecture parameters: model, dataset, random_w
    training regime: baseline (if not baseline: use cluster assignment with sampling formula to finetune)
    clustering parameters: atoms, entropy, metric, sparsity
    dataset loading parameters: dataset, dataset_params

    """

    args = parser.parse_args()

    if args.baseline and args.cluster_assignment_file != "":
        print("If you're using Baseline, then dont provide assignments")
        sys.exit(1)


    dataset = get_dataset(args)
    sampling_ratios = generate_sampling_ratios(args, len(dataset['train']), args.cluster_assignment_file, args.sampling_formula)
    loaders, sizes = get_initial_loaders(dataset, sampling_ratios)
    
    # Use new model architecture function
    model = get_model_architecture(args)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.epochs, gamma=0.1)

    model = train_model(device, model, loaders, sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)
    
    # Use new save path function
    save_path = get_model_save_path(args)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)