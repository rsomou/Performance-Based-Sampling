import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import zipfile
import requests
from io import BytesIO
from sklearn import preprocessing
import keras
from torch.utils.data import DataLoader
import csv
import time
from scipy.spatial import KDTree

import argparse
from data.Dataset import get_dataset, ImageDataset, ImageDatasetSampled, transform, custom_collate_fn, generate_train_dataset
from utils.models import (
    get_model_architecture, 
    get_model_save_path, 
    get_dataset_root
)


from data.Dataset import ImageDataset, transform, custom_collate_fn
from utils.adv_utils import nnk_clustering

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--dataset_params", nargs='+', type=str, default=[], help="corruption type")

    parser.add_argument("--features_path", type=str, default="", help="path to feature csv") 
    parser.add_argument("--model_file", type=str, default="", help="model file path to load directly")

    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--random_w", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="resnet-s")


    #clustering parameters
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--atoms", type=int, default=20)
    parser.add_argument("--sparsity", type=int, default=5)
    parser.add_argument("--metric", type=str, default="error")
    parser.add_argument("--entropy", type=float, default=0)

    """

    getting model architecture parameters: model, dataset, random_w
    model loading parameters: model_file
    feature loading parameters: features_path
    clustering parameters: atoms, entropy, metric, sparsity, epochs
    dataset loading parameters: dataset, dataset_params

    """
    args = parser.parse_args()

    # potentially implement branching when low level or high level features is developed

    dataset = get_dataset(args)

    emb = []
    with open(args.features_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert each row to a list of floats (adjust if data is not numeric)
            emb.append([float(x) for x in row])

    # Get model and load weights
    model = get_model_architecture(args)

    save_path = ""
    if args.model_file == "":
        save_path = get_model_save_path(args)
    else:
        save_path = args.model_file


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(save_path, map_location = device)
    model.load_state_dict(state_dict)


    nnk_clustering(emb, dataset['train'], model, args)


