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

#haven't touched
def generate_features(training_data, val_data, bbox, features, ds): 
  train_vecs = []

  fts = ["contrast", "color", "size", "CLIP"]
  #if all(ft not in features for ft in fts):
    #raise ValueError("Invalid feature(s)- please select atleast 1 out of [CLIP, contrast, color, size]")
  
  if "contrast" in features:
    if not os.path.exists(featuresroot + "/" + ds + "_contrast_t"):
      contrast_results_t = np.zeros((len(training_data),1))
      # contrast_results_v = np.zeros((len(val_data),1))

      for i in range(len(training_data)):
        print(np.array(training_data[i]))
        contrast_results_t[i] = contrast(np.array(training_data[i]))
      max_contrast_t = max(contrast_results_t)
      for i in range(len(contrast_results_t)):
        contrast_results_t[i] = contrast_results_t[i]/max_contrast_t

      # for i in range(len(val_data)):
        # contrast_results_v[i] = contrast(np.array(val_data[i]))
      # max_contrast_v = max(contrast_results_v)
      # for i in range(len(contrast_results_v)):
        # contrast_results_v[i] = contrast_results_v[i]/max_contrast_v

      np.savetxt(featuresroot + "/" + ds + "_contrast_t", contrast_results_t, delimiter=",")
      # np.savetxt(featuresroot + "/" + ds + "_contrast_v", contrast_results_v, delimiter=",")
    else:
      contrast_results_t = np.loadtxt(featuresroot + "/" + ds + "_contrast_t", delimiter=",")
      # contrast_results_v = np.loadtxt(featuresroot + "/" + ds + "_contrast_v", delimiter=",")
    if len(train_vecs) == 0:
      train_vecs = contrast_results_t
    else:
      train_vecs = np.concatenate((train_vecs, contrast_results_t), axis=1)
    # val_vecs = np.concatenate((val_vecs, contrast_results_v), axis=1)
    print("Computed contrasts")

  if "color" in features:
    if not os.path.exists(featuresroot + "/" + ds + "_color_t"):
      
      color_dict = {
        "#000000": "black",  # Black
        "#FFFFFF": "white",  # White
        "#808080": "gray",   # Gray
        "#FF0000": "red",    # Red
        "#FFA500": "orange", # Orange
        "#FFFF00": "yellow", # Yellow
        "#008000": "green",  # Green
        "#0000FF": "blue",   # Blue
        "#800080": "purple", # Purple
        "#FFC0CB": "pink",   # Pink
        "#A52A2A": "brown"   # Brown
      }
      
      color_code = {0: "black", 1: "white", 2: "gray", 3: "red", 4: "orange", 5: "yellow", 6: "green", 7: "blue", 8: "purple", 9: "pink", 10: "brown"}
      css3_db = color_dict
      names = []
      rgb_values = []
      
      kdt_db = KDTree(rgb_values)  
      
      for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

      color_results_t = np.zeros((len(training_data),1))
      # color_results_v = np.zeros((len(val_data),11))

      for i in range(len(training_data)):
        color_results_t[i] = color_distrib(np.array(training_data[i]))
      max_color_t = max(color_results_t)
      for i in range(len(color_results_t)):
        color_results_t[i] = color_results_t[i]/max_color_t

      # for i in range(len(val_data)):
        # color_results_v[i] = color_distrib(np.array(val_data[i]))
      # max_color_v = max(color_results_v)
      # for i in range(len(color_results_v)):
        # colorresults_v[i] = color_results_v[i]/max_color_v

      np.savetxt(featuresroot + "/" + ds + "_color_t", color_results_t, delimiter=",")
      # np.savetxt(featuresroot + "/" + ds + "_color_v", color_results_v, delimiter=",")
    else:
      color_results_t = np.loadtxt(featuresroot + "/" + ds + "_color_t", delimiter=",")
      # color_results_v = np.loadtxt(featuresroot + "/" + ds + "_color_v", delimiter=",")
    if len(train_vecs) == 0:
      train_vecs = color_results_t
    else:
      train_vecs = np.concatenate((train_vecs, color_results_t), axis=1)
    # val_vecs = np.concatenate((val_vecs, color_results_v), axis=1)
    print("Computed color distributions")

  if "size" in features and ds == "imagenet":
    if not os.path.exists(featuresroot + "/" + ds + "_size_t"):
      if os.path.exists(featuresroot + "/" + ds + "_size_t"):
        bbox_results_t = np.zeros((len(training_data),1))
        # bbox_results_v = np.zeros((len(val_data),11))

      for i in range(len(training_data)):
        bbox_results_t[i] = bbox_area(bbox_data["train"][i])
      max_size_t = max(bbox_results_t)
      for i in range(len(bbox_results_t)):
        bbox_results_t[i] = bbox_results_t[i]/max_size_t

      # for i in range(len(val_data)):
        # bbox_results_v[i] = bbox_area(bbox_data["val"][i])
      # max_size_v = max(bbox_results_v)
      # for i in range(len(bbox_results_v)):
        # bbox_results_v[i] = bbox_results_v[i]/max_size_v

      np.savetxt(featuresroot + "/" + ds + "_size_t", bbox_results_t, delimiter=",")
      # np.savetxt(featuresroot + "/" + ds + "_size_v", bbox_results_v, delimiter=",")
    else:
      bbox_results_t = np.loadtxt(featuresroot + "/" + ds + "_size_t", delimiter=",")
      # bbox_results_v = np.loadtxt(featuresroot + "/" + ds + "_size_v", delimiter=",")
    train_vecs = np.concatenate((train_vecs, bbox_results_t), axis=1)
    # val_vecs = np.concatenate((val_vecs, bbox_results_v), axis=1)

  return train_vecs
  

def mapping(emm_matrix, data):
    assert len(data) == len(emm_matrix), (
        f"Dataset length ({len(data)}) must match num of embedding vectors ({len(emm_matrix)})"
    )
    mp = []
    for i, (_,label) in enumerate(data):
      mp.append({'vector': emm_matrix[i],'label': label})
    return mp
      

def nnk_clustering(emm_matrix, data, model, ds, epochs, atoms, top_k, metric, ep):
    """
    Perform NNK clustering within each class.
    
    Args:
        emm_matrix: numpy array of embeddings (num_samples x embedding_dim)
        data: list of dictionaries containing 'image' and 'label' keys (pre provided split (train or valid or test))
        ds: dataset name (string)
        epochs: number of clustering epochs
        sparsity: sparsity parameter (not used)
        atoms: number of clusters per class
        top_k: k nearest neighbors
        ep: epsilon parameter
    """
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
    cluster_file =  CLUSTERS_DIR + "/" + ds + "-assignments.csv"
    os.makedirs(os.path.dirname(cluster_file), exist_ok=True)
    if os.path.exists(cluster_file):
        os.remove(cluster_file)
    
    assignments = []
    
    # Cluster each class separately
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


        a = 0.5  # Tunable parameter
        sampling_ratios = []
        for entry in asgn_mp:
            class_idx = entry['class_idx']
            cluster_idx = entry['cluster_idx']
            acc = entry['acc']
            acc_mean = class_to_avg_acc[class_idx]

            # Sampling ratio formula
            sampling_ratio = min(2,np.exp((acc_mean - acc) / a))
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
    
    for class_id, clusters in class_clusters.items():
        cluster_accuracies = []
        for cluster_id, indices in clusters.items():
            if len(indices) > 0:
                acc = eval_cluster(model, indices, data)
                cluster_accuracies.append(acc)
        
        if len(cluster_accuracies) > 1:  # Need at least 2 clusters to compute variance
            variance = np.var(cluster_accuracies)
            class_variances.append(variance)
            print(f"Class {class_id} variance: {variance:.4f}")
    
    mean_variance = np.mean(class_variances) if class_variances else 0
    
    return mean_variance, class_variances