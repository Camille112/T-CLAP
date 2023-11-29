"""
This version of the code is developed by anshs@gatech.edu
"""

import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils
import random

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
import torch
from datasets import load_dataset

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

#import positional_encoder as pe
import time
from clap_dds import AudioTextDataset
import clap_ds as ds
import clap_dds as dds
import clap_wrap

import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load dataset

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def set_top_two_max_to_one(array):
    # Flatten the array to ensure it's one-dimensional
    flattened_array = array.flatten()

    # Find the indices of the two largest values
    indices_of_max_values = np.argpartition(flattened_array, -2)[-2:]

    # Create a new array of zeros with the same shape as the flattened array
    result_array = np.zeros_like(flattened_array)

    # Set the top two positions to 1
    result_array[indices_of_max_values] = 1

    return np.array(result_array)


filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset,test_dataset = dds.split_dataset(dataset,test_size=0.2, random_state=42)

batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

# Load your configuration file
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)

# Path to your downloaded file
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'
# Reading the file and storing each line in a list
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

prompt = 'this is a sound of '
y = [prompt + x for x in words_list]


def one_hot_encode(text, words_list=words_list):
    # Create a one-hot encoded vector as a numpy array
    one_hot_vector = np.zeros((1, len(words_list)))

    # Iterate over each word or phrase in words_list
    for i, word_or_phrase in enumerate(words_list):
        # Check if the word or phrase is in the text
        if word_or_phrase in text:
            one_hot_vector[0, i] = 1

    return np.array(one_hot_vector[0])

def compare_vectors_and_score(vec1, vec2, points_per_match=0.5):
    # Ensure both vectors have the same length
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    # Initialize the score
    score = 0

    # Iterate over the vectors and check if the ones match
    for i in range(len(vec1)):
        if vec1[i] == 1 and vec2[i] == 1:
            score += points_per_match
    
    if score ==1:
        score = 1
    else:
        score = 0

    return score

weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
#weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
#weights_path = '/Users/anshumansinha/Downloads/HW4_/model_epoch_100_1.51.pth'
#weights_path = '/Users/anshumansinha/Downloads/HW4_/model_epoch_82_0.19.pth' # 

clap_model = CLAP(weights_path, version = '2022', use_cuda=False)

print('*'*20)
print('y', y)

text_embeddings_f = clap_model.get_text_embeddings(y)
print(text_embeddings_f.shape)

y_preds, y_labels = [], []
total_s = 0

for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r,audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_

    elements = [text_b_f,text_b_r]
    weights = [0.5, 0.5]
    picked_element = random.choices(elements, weights, k=1)[0]
    one_hot_target_t = picked_element[0]


    if one_hot_target_t == text_b_f[0]:
        aud_s = [audio_b_f[0]]
    else:
        aud_s = [audio_b_r[0]]

    one_hot_target_t = text_b_w[0]
    aud_s = [audio_b_w[0]]

    # get the 1 hot class for this text
    one_hot_target = one_hot_encode(one_hot_target_t)
    audio_embeddings = clap_model.get_audio_embeddings(aud_s, resample=False)

    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()

    total_s+= compare_vectors_and_score(one_hot_target,set_top_two_max_to_one(y_pred))



acc = total_s/len(test_loader)
print(total_s)
print(len(test_loader))
print(acc)
print('ESC50 Accuracy {}'.format(acc))

"""
The output:

ESC50 Accuracy: 93.9%

"""
