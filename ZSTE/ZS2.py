"""
This script processes audio-text datasets using the CLAP model for classification.
Developed by anshs@gatech.edu
"""

import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import random
import torchvision
import torch.nn as nn
import torchvision.models as models
from clap import CLAP as CLAPS
from msclap import CLAP
import warnings
import yaml
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import pandas as pd
import os
from datasets import load_dataset
import math
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
from clap_dds import AudioTextDataset
import clap_ds as ds
import clap_dds as dds
import clap_wrap
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# Load dataset configuration
def load_config(file_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def set_top_two_max_to_one(array: np.ndarray) -> np.ndarray:
    """
    Set the two highest values in an array to 1, and the rest to 0.
    
    Args:
        array (np.ndarray): Input array to be processed.
    
    Returns:
        np.ndarray: Array with the two highest values set to 1.
    """
    # Flatten the array to ensure it's one-dimensional
    flattened_array = array.flatten()

    # Find the indices of the two largest values
    indices_of_max_values = np.argpartition(flattened_array, -2)[-2:]

    # Create a new array of zeros with the same shape as the flattened array
    result_array = np.zeros_like(flattened_array)

    # Set the top two positions to 1
    result_array[indices_of_max_values] = 1

    return result_array

# Path to the dataset JSON file
filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset, test_size=0.2, random_state=42)

batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

# Load your configuration file
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)

# Path to your downloaded file with words
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'
# Reading the file and storing each line in a list
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

prompt = 'this is a sound of '
y = [prompt + x for x in words_list]

def one_hot_encode(text: str, words_list=words_list) -> np.ndarray:
    """
    One-hot encode a given text based on a list of words/phrases.
    
    Args:
        text (str): The input text to be encoded.
        words_list (list): List of words/phrases to check against.
    
    Returns:
        np.ndarray: One-hot encoded vector.
    """
    # Create a one-hot encoded vector as a numpy array
    one_hot_vector = np.zeros((1, len(words_list)))

    # Iterate over each word or phrase in words_list
    for i, word_or_phrase in enumerate(words_list):
        # Check if the word or phrase is in the text
        if word_or_phrase in text:
            one_hot_vector[0, i] = 1

    return one_hot_vector[0]

def compare_vectors_and_score(vec1: np.ndarray, vec2: np.ndarray, points_per_match=0.5) -> int:
    """
    Compare two vectors and score based on the number of matches.
    
    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.
        points_per_match (float): Points awarded per match.
    
    Returns:
        int: Score based on matching elements.
    """
    # Ensure both vectors have the same length
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    # Initialize the score
    score = 0

    # Iterate over the vectors and check if the ones match
    for i in range(len(vec1)):
        if vec1[i] == 1 and vec2[i] == 1:
            score += points_per_match
    
    # Adjust score based on the matches
    if score == 1:
        score = 1
    else:
        score = 0

    return score

# Path to the CLAP model weights
weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
clap_model = CLAP(weights_path, version='2022', use_cuda=False)

print('*'*20)
print('y', y)

# Generate text embeddings using the CLAP model
text_embeddings_f = clap_model.get_text_embeddings(y)
print(text_embeddings_f.shape)

y_preds, y_labels = [], []
total_s = 0

# Iterate over the test data loader
for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r, audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_

    elements = [text_b_f, text_b_r]
    weights = [0.5, 0.5]
    picked_element = random.choices(elements, weights, k=1)[0]
    one_hot_target_t = picked_element[0]

    if one_hot_target_t == text_b_f[0]:
        aud_s = [audio_b_f[0]]
    else:
        aud_s = [audio_b_r[0]]

    one_hot_target_t = text_b_w[0]
    aud_s = [audio_b_w[0]]

    # Get the one-hot class for this text
    one_hot_target = one_hot_encode(one_hot_target_t)
    audio_embeddings = clap_model.get_audio_embeddings(aud_s, resample=False)

    # Compute similarity between audio and text embeddings
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()

    total_s += compare_vectors_and_score(one_hot_target, set_top_two_max_to_one(y_pred))

# Calculate accuracy
acc = total_s / len(test_loader)
print(total_s)
print(len(test_loader))
print(acc)
print('ESC50 Accuracy {}'.format(acc))
