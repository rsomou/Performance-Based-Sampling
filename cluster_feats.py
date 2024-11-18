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

from utils.features import contrast, color_distrib, bbox_area, hex_to_rgb
from data.Dataset import ImageDataset, transform, custom_collate_fn
from utils.adv_utils import nnk_clustering

seed = 145
metric = 'error' # anomaly detection metric


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default="cifar100")
  #parser.add_argument("--low_lvl_features", nargs="+", type=str, default=["contrast", "color", "size"], help="Select features contrast, color, size")
  parser.add_argument("--features_path", type=str, default="./CLIP_cifar100_train.csv", help="path to feature csv") 
  parser.add_argument("--model_file", type=str, default="", help="model file path to load directly")
  parser.add_argument("--baseline", action="store_true", default=False) 
  parser.add_argument("--epochs", type=int, default=7)
  parser.add_argument("--model", type=str, default="resnet-s")

  parser.add_argument("--atoms", type=int, default=20)
  parser.add_argument("--sparsity", type=int, default=5)
  parser.add_argument("--metric", type=str, default="error")
  parser.add_argument("--entropy", type=int, default=0)
  parser.add_argument("--clusters_dir", type=str, default="./clusters")
  args = parser.parse_args()

  # potentially implement branching when you have low level or high level features within this py file

  dataset, bbox_data, counts = get_dataset(args)

  emb = []
  with open(args.features_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # Convert each row to a list of floats (adjust if data is not numeric)
        emb.append([float(x) for x in row])

  # Get model and load weights
  model = get_model_architecture(args.model, args.dataset)

  save_path = ""
  if args.model_file == "":
      save_path = get_model_save_path(args)
  else:
      save_path = args.model_file

  model.load_state_dict(torch.load(save_path))


  nnk_clustering(emb, dataset['train'], model, args.dataset, args.epochs, args.atoms, args.sparsity, args.metric, args.entropy)
  

