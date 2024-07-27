"""
This version of the code is developed by anshs@gatech.edu
"""

import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils
import torch
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
import torch.nn as nn
from datasets import load_dataset
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import time
from clap_dds import AudioTextDataset
import clap_ds as ds
import clap_dds as dds
import clap_wrap
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def load_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML data as a dictionary.
    """
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def torch_device_select(gpu):
    """
    Select the appropriate device (CPU or GPU) based on availability and user preference.

    Args:
        gpu (bool): If True, prefer GPU over CPU.

    Returns:
        str: 'cuda' if GPU is available and preferred, otherwise 'cpu'.
    """
    if torch.cuda.is_available() and not gpu:
        warnings.warn("GPU is available but not used.")
        return 'cpu'
    elif not torch.cuda.is_available() and gpu:
        warnings.warn("GPU is not available but set to used. Using CPU.")
        return 'cpu'
    elif torch.cuda.is_available() and gpu:
        return 'cuda'
    else:
        return 'cpu'

# Load dataset
filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset)
batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

# Save path for models
save_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/'

# Reading the file and storing each line in a list
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

# Create prompts for text embeddings
prompt1 = 'In this concatenated sound, the first sound is '
prompt2 = 'In this concatenated sound, the second sound is '
y1 = [prompt1 + x for x in words_list]
y2 = [prompt2 + x for x in words_list]
y = y1 + y2

print(y)

def compare_vectors_and_score(vec1, vec2, points_per_match=0.5):
    """
    Compare two vectors and calculate a score based on matching elements.

    Args:
        vec1 (np.array): First binary vector.
        vec2 (np.array): Second binary vector.
        points_per_match (float): Points awarded for each matching '1' in the vectors.

    Returns:
        float: The score based on the number of matching '1's in both vectors.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    score = 0
    for i in range(len(vec1)):
        if vec1[i] == 1 and vec2[i] == 1:
            score += points_per_match
    
    if score == 1:
        score = 1
    else:
        score = 0

    return score

def set_top_two_max_to_one(array):
    """
    Set the two highest values in the array to one, and the rest to zero.

    Args:
        array (np.array): Input array.

    Returns:
        np.array: Array with two highest values set to one and others to zero.
    """
    flattened_array = array.flatten()
    indices_of_max_values = np.argpartition(flattened_array, -2)[-2:]
    result_array = np.zeros_like(flattened_array)
    result_array[indices_of_max_values] = 1

    return np.array(result_array).reshape(1, -1)

def one_hot_for_any_word(text, text2, sentence_list=y1):
    """
    Generate a one-hot encoding for any matching words in the provided text.

    Args:
        text (str): Input text for comparison.
        text2 (str): Second input text for comparison.
        sentence_list (list): List of sentences to compare against.

    Returns:
        np.array: One-hot encoded vector indicating matching words.
    """
    text_words = set(text.split())
    last_two_words_list = [set(sentence.split()[-2:]) for sentence in sentence_list]
    vector = [0] * 2 * len(sentence_list)

    for i, last_words in enumerate(last_two_words_list):
        if text_words.intersection(last_words):
            vector[i] = 1

    text_words = set(text2.split())
    last_two_words_list = [set(sentence.split()[-2:]) for sentence in sentence_list]

    for i, last_words in enumerate(last_two_words_list):
        if text_words.intersection(last_words):
            vector[i] = 1

    return np.array(vector).reshape(1, -1)

# Load and initialize CLAP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)
weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
clap_model = CLAP(weights_path, version='2022', use_cuda=False)

# Computing text embeddings
print('*' * 20)
print('y', y)
text_embeddings_f = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
total_s = 0

for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']
    audio_a, audio_b, audio_b_f, audio_b_r, audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_

    one_hot_target = one_hot_for_any_word(text_a[0], text_b[0])
    audio_embeddings = clap_model.get_audio_embeddings([audio_b_f[0]], resample=False)

    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()

    total_s += compare_vectors_and_score(one_hot_target[0], set_top_two_max_to_one(y_pred)[0])

acc = total_s / len(test_loader)
print(total_s)
print(len(test_loader))
print(acc)
print('ESC50 Accuracy {}'.format(acc))
