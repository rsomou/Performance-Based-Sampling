from collections import defaultdict
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
from torch.utils.data import DataLoader, Subset
import csv
import time
from scipy.spatial import KDTree
from PIL import Image

import argparse

from data.Dataset import get_dataset, ImageDataset, ImageDatasetSampled, transform, custom_collate_fn, generate_train_dataset
from utils.models import (
    get_model_architecture, 
    get_model_save_path, 
    get_dataset_root, 
    MODEL_DIR
)

from utils.nnk import NNKMU
from utils.features import contrast, color_distrib, bbox_area, hex_to_rgb


CLUSTERS_DIR = "./clustering/"

color_dict = {
    "#000000": "black",
    "#FFFFFF": "white",
    "#808080": "gray",
    "#FF0000": "red",
    "#FFA500": "orange",
    "#FFFF00": "yellow",
    "#008000": "green",
    "#0000FF": "blue",
    "#800080": "purple",
    "#FFC0CB": "pink",
    "#A52A2A": "brown"
}

names = list(color_dict.values())
rgb_values = [hex_to_rgb(hex_color) for hex_color in color_dict.keys()]

kdt_db = KDTree(rgb_values)

def contrast(image):
    r_coeff = 0.2989
    g_coeff = 0.5870
    b_coeff = 0.1140
    gray_image = (image[..., 0] * r_coeff +
                  image[..., 1] * g_coeff +
                  image[..., 2] * b_coeff)
    return gray_image.std()

def convert_rgb_to_names(rgb_array):
    _, indices = kdt_db.query(rgb_array)
    return np.array(names)[indices]

def color_distrib(image):

    if image.size == 4096:  # Grayscale image check
        # Optional: Convert grayscale to RGB (simple replication across channels)
        reshaped_image = np.stack((image,)*3, axis=-1).reshape(-1, 3)
    else:
        # Reshape assuming it's already an RGB image
        reshaped_image = np.reshape(image, (-1, 3))

    # Convert RGB values to color names using vectorized operation
    color_names = convert_rgb_to_names(reshaped_image)

    # Calculate the frequency distribution of color names
    unique, counts = np.unique(color_names, return_counts=True)
    color_distribution = dict(zip(unique, counts))

    # Create a distribution array
    distrib = np.array([color_distribution.get(color, 0) for color in names])

    # Normalize the distribution to sum to 1
    distrib_normalized = distrib / np.sum(distrib)
    return distrib_normalized

def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviations
    xmax = max(
        abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta))
    )
    xmax = np.ceil(max(1, xmax))
    ymax = max(
        abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta))
    )
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


#haven't touched
def generate_features(training_data, ds, output):
    """
    Generate a matrix of low-level features (color distribution, contrast, Gabor)
    for each image in the specified dataset split (train or valid). 
    Each row is a flattened feature vector, optionally followed by a label.

    :param training_data: A dict like {"train": [...], "valid": [...]}, 
                          each containing a list of {"image": PIL.Image, "label": int}.
    :param ds: A string, either "train" or "valid", indicating which split to use.
    :param output: A file path to write the features out row by row.
    """

    with open(output, "w") as f:
        writer = csv.writer(f)
        for item in training_data[ds]:
            # 1. Get the PIL image and convert to numpy
            pil_img = item["image"]
            np_img  = np.array(pil_img)  # shape = (H, W, 3) if RGB

            # 2. Compute the color distribution (returns e.g. a length-11 vector)
            cdist = color_distrib(np_img)  # Already normalized

            # 3. Compute the contrast (single scalar)
            contr = contrast(np_img)

            # 4. Generate a Gabor filter (example parameters)
            #    Typically, you'd convolve this with the image or compute some statistic;
            #    for demonstration, we'll just flatten the filter itself.
            gb_filter = gabor(
                sigma=1.0,   # std. deviation
                theta=0,     # orientation in radians
                Lambda=10.0, # wavelength
                psi=0,       # phase offset
                gamma=0.5    # aspect ratio
            )
            gb_flat = gb_filter.flatten()  # Flatten the 2D kernel

            # 5. Concatenate all features into one row vector
            #    (color distribution, contrast, gabor filter)

            writer.writerow(np.concatenate([cdist, [contr], gb_flat]).tolist())

    print(f"Features saved to {output}")
  

def mapping(emm_matrix, data):
    assert len(data) == len(emm_matrix), (
        f"Dataset length ({len(data)}) must match num of embedding vectors ({len(emm_matrix)})"
    )
    mp = []
    for i, (_,label) in enumerate(data):
      mp.append({'vector': emm_matrix[i],'label': label})
    return mp
      

def nnk_clustering(emm_matrix, data, model, args):
    
    """
    Perform NNK clustering within each class.
    
    Args:
        emm_matrix: numpy array of embeddings (num_samples x embedding_dim)
        data: list of dictionaries containing 'image' and 'label' keys (pre provided split (train or valid or test))

        -- Config Arguements

        ds: dataset name (string)
        epochs: number of clustering epochs
        sparsity: sparsity parameter (not used)
        atoms: number of clusters per class
        top_k: k nearest neighbors
        ep: epsilon parameter
    """

    model_name, ds_name, epochs, atoms, top_k, metric, ep = args.model, args.dataset, args.epochs, args.atoms, args.sparsity, args.metric, args.entropy
    start_time = time.time()
    
    # Create class-wise indices and embeddings
    class_indices = defaultdict(list)
    class_embeddings = defaultdict(list)
    
    # Group data by class
    for idx, (embedding, sample) in enumerate(zip(emm_matrix, data)):
        label = sample['label']
        class_indices[label].append(idx)
        class_embeddings[label].append(embedding)
    
    # Prepare output file
    cluster_file =  CLUSTERS_DIR + f"{ds_name}-{model_name}-{ep}-{atoms}-{top_k}-assignments.csv"
    os.makedirs(os.path.dirname(cluster_file), exist_ok=True)
    if os.path.exists(cluster_file):
        os.remove(cluster_file)
    
    with open(cluster_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for class_idx in sorted(class_indices.keys()):
            print(f"Clustering class {class_idx}...")
            
            # Get class-specific data
            indices = class_indices[class_idx]
            embeddings = np.array(class_embeddings[class_idx], dtype=np.float32)
            
            # Skip if class has too few samples
            if len(indices) < atoms:
                print(f"Skipping class {class_idx}: too few samples")
                continue
            
            # Perform clustering
            nnkmodel = NNKMU(num_epochs=epochs, metric=metric, n_components=atoms, top_k=top_k, ep=ep, weighted=False, num_warmup=2)
            nnkmodel.fit(embeddings)
            
            # Get cluster assignments
            codes = nnkmodel.get_codes(embeddings)
            cluster_assignments = torch.argmax(codes, dim=1).numpy()

            cluster_members = defaultdict(list)
            for local_idx, global_idx in enumerate(indices):
              cluster = cluster_assignments[local_idx]
              cluster_members[cluster].append(global_idx)

            cluster_scores = {}
            for cluster_idx, member_indicies in cluster_members.items():
                if(len(member_indicies)>0):
                    acc = eval_cluster(model, member_indicies, data)
                    cluster_scores[cluster_idx] = acc
                    print(f"Class {class_idx}, Cluster: {cluster_idx}, Acc: {acc}")

            for local_idx, global_idx in enumerate(indices):
               cluster = cluster_assignments[local_idx]
               writer.writerow([global_idx, cluster_scores[cluster], cluster, class_idx])
            
            # Print progress
            print(f"Processed {len(indices)} samples in class {class_idx}")

    end_time = time.time()
    print(f"Total clustering time: {end_time - start_time:.2f} seconds")
    return cluster_file


  
def eval_cluster(model, dataset_indices, dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = len(dataset_indices)

    cluster_data = []
    for idx in dataset_indices:
       cluster_data.append(dataset[idx])

    cluster_dataset = ImageDataset({'cluster': cluster_data}, 'cluster', transform=transform)
    cluster_loader = DataLoader(cluster_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for images, labels in cluster_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def generate_sampling_ratios(args, ds_len, assignments , method="max_err"):
   
    if args.baseline or args.cluster_assignment_file == "":
        print("Default Sampling Ratios Loaded")
        sampling_ratios = [1] * ds_len
        return sampling_ratios
    
    asgn_mp = [{} for _ in range(ds_len)]

    with open (assignments, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            global_idx, acc, cluster_idx, class_idx = int(row[0]), float(row[1]), int(row[2]), int(row[3])
            asgn_mp[global_idx] = {'acc': acc, 'cluster_idx': cluster_idx, 'class_idx': class_idx}


    if method == "max_err":
        class_to_cluster_errors = {}
        class_to_max_error = {}

        for entry in asgn_mp:
            class_idx = entry['class_idx']
            cluster_idx = entry['cluster_idx']
            error = 1 - entry['acc'] 

            # Group errors by class and cluster
            if class_idx not in class_to_cluster_errors:
                class_to_cluster_errors[class_idx] = {}
            if cluster_idx not in class_to_cluster_errors[class_idx]:
                class_to_cluster_errors[class_idx][cluster_idx] = error
            else:
                class_to_cluster_errors[class_idx][cluster_idx] = error

            # Update max error for the class
            if class_idx not in class_to_max_error:
                class_to_max_error[class_idx] = error
            else:
                class_to_max_error[class_idx] = max(class_to_max_error[class_idx], error)

        # Compute sampling ratios
        sampling_ratios = []
        for entry in asgn_mp:
            class_idx = entry['class_idx']
            cluster_idx = entry['cluster_idx']
            cluster_error = class_to_cluster_errors[class_idx][cluster_idx]
            max_error = class_to_max_error[class_idx]

            # Sampling ratio formula
            sampling_ratio = args.epsilon + (cluster_error / (max_error if max_error > 0 else 1))
            sampling_ratios.append(sampling_ratio)

        return sampling_ratios
    
    if method == "exp_dis":
        class_to_cluster_acc = {}
        class_to_avg_acc = {}

        for entry in asgn_mp:
            class_idx = entry['class_idx']
            cluster_idx = entry['cluster_idx']
            acc = entry['acc'] 

            # Group errors by class and cluster
            if class_idx not in class_to_cluster_acc:
                class_to_cluster_acc[class_idx] = {}
            if cluster_idx not in class_to_cluster_acc[class_idx]:
                class_to_cluster_acc[class_idx][cluster_idx] = acc
            else:
                class_to_cluster_acc[class_idx][cluster_idx] = acc

        for class_idx, clusters in class_to_cluster_acc.items():
            avg_acc = np.mean(list(clusters.values()))  # Average accuracy of all clusters in the class
            class_to_avg_acc[class_idx] = avg_acc


        a = 0.8 # Tunable parameter
        sampling_ratios = []
        for entry in asgn_mp:
            class_idx = entry['class_idx']
            cluster_idx = entry['cluster_idx']
            acc = entry['acc']
            acc_mean = class_to_avg_acc[class_idx]

            # Sampling ratio formula
            sampling_ratio = np.clip(np.exp((acc_mean - acc) / a),1,2.75)
            sampling_ratios.append(sampling_ratio)

        return sampling_ratios
            
    
    raise ValueError(f"Method '{method}' is not implemented.")


def evaluate_cluster_variance(cluster_file, model, data):
    """
    Calculate mean of class variances in cluster performances
    
    Args:
        cluster_file: Path to CSV containing cluster assignments
        model: Model to evaluate cluster performances
        data: Dataset split containing images and labels
        
    Returns mean of all class variances

    """
    # Read assignments and group by class
    class_clusters = defaultdict(lambda: defaultdict(list))
    with open(cluster_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            global_idx = int(row[0])
            cluster_id = int(row[2])
            class_id = int(row[3])
            class_clusters[class_id][cluster_id].append(global_idx)
    
    # Calculate variance for each class
    class_variances = []
    min_class_acc = []
    cluster_stats = []

    for class_id, clusters in class_clusters.items():
        cluster_accuracies = []
        min_acc = 101
        stats = {}
        for cluster_id, indices in clusters.items():
            if len(indices) > 0:
                acc = eval_cluster(model, indices, data)
                stats[cluster_id] = (len(indices), acc)
                min_acc = min(min_acc, acc)
                cluster_accuracies.append(acc)
        
        if len(cluster_accuracies) > 1:  # Need at least 2 clusters to compute variance
            variance = np.var(cluster_accuracies)
            class_variances.append(variance)
            print(f"Class {class_id} variance: {variance:.4f}")
        
        min_class_acc.append(min_acc)
        cluster_stats.append(stats)
    
    mean_variance = np.mean(class_variances) if class_variances else 0
    
    return mean_variance, class_variances, min_class_acc, cluster_stats