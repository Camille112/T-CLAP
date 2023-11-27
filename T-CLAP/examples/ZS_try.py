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

def torch_device_select(gpu):
    # check GPU availability & return device type
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


filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'

dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset,test_dataset = dds.split_dataset(dataset)

batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
save_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/'

# Path to your downloaded file
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'
root_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/'

# Reading the file and storing each line in a list
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

prompt = 'this is a sound of '
y = [prompt + x for x in words_list]


print('words_list',words_list)


def one_hot_encode(text, words_list=words_list):
    # Create a one-hot encoded vector as a numpy array
    one_hot_vector = np.zeros((1, len(words_list)))

    # Check if any word or phrase from words_list is at the beginning of the text
    for i, word_or_phrase in enumerate(words_list):
        if text.startswith(word_or_phrase):
            one_hot_vector[0, i] = 1
            break  # Stop the loop if a match is found

    return np.array(one_hot_vector)


# Load and initialize CLAP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your configuration file
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)

'''
weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
clap_model = CLAP(weights_path, version = '2022', use_cuda=False)
'''

#weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
# Load the state dictionary into the model
#clap_model.load_state_dict(state_dict)

#weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
clap_model = CLAP(weights_path, version = '2022', use_cuda=False)
# Load the state dictionary into the model

print('*'*20)
# Computing text embeddings
print('y',y)

text_embeddings_f = clap_model.get_text_embeddings(y)
print(text_embeddings_f.shape)

# Computing audio embeddings
y_preds, y_labels = [], []

for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r,audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_
    one_hot_target = text_a[0]

    # get the 1 hot class for this text
    one_hot_target = one_hot_encode(one_hot_target)
    
    #print('*'*20)
    #print(one_hot_target)

    max_value = np.amax(one_hot_target, axis=1)
    max_index = np.argmax(one_hot_target, axis=1)
    
    #print('one_hot_target',max_value)
    #print('one_hot_target',max_index)
    #print(text_a[0])
    #print(one_hot_target)

    audio_embeddings = clap_model.get_audio_embeddings([audio_a[0]], resample=True)
    #print(audio_embeddings)
    
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()

    y_preds.append(y_pred)
    y_labels.append(one_hot_target)


print(y_labels[0].shape)
print(y_preds[0].shape)

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)

print(y_labels.shape)
print(y_preds.shape)

acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))

"""
The output:

ESC50 Accuracy: 93.9%

"""
