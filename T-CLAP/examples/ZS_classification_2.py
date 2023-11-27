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
from clap_dds_1 import AudioTextDataset
import clap_ds as ds
import clap_dds_1 as dds
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


filepath = '/Users/anshumansinha/Downloads/HW3/esc50-temporal-pairs-reduced2.json'
#filepath = '/Users/anshumansinha/Downloads/HW3/esc50-temporal-pairs-reduced2_0.json'

dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset,test_dataset = dds.split_dataset(dataset)

batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
save_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/'

# Path to your downloaded file
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'

# Reading the file and storing each line in a list
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

prompt = 'this is a sound of '
y = [prompt + x for x in words_list]

def one_hot_encode(text, words_list=words_list):
    # Split the text to get the first word
    first_word = text.split()[0]

    # Create a one-hot encoded vector as a numpy array
    one_hot_vector = np.zeros((1, len(words_list)))

    # Check if the first word is in the words_list
    if first_word in words_list:
        index = words_list.index(first_word)
        one_hot_vector[0, index] = 1

    return np.array(one_hot_vector)

# Load and initialize CLAP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your configuration file
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)

clap_model = CLAPS(
            audioenc_name=config['audioenc_name'],
            sample_rate=config['sampling_rate'],  # 'sample_rate' seems to be the correct parameter name
            window_size=config['window_size'],
            hop_size=config['hop_size'],
            mel_bins=config['mel_bins'],
            fmin=config['fmin'],
            fmax=config['fmax'],
            classes_num=config['num_classes'],  # It seems 'classes_num' is the correct parameter name
            out_emb=config['out_emb'],
            text_model=config['text_model'],
            transformer_embed_dim=config['transformer_embed_dim'],
            d_proj=config['d_proj'])

# Load the state dictionary
state_dict = torch.load('/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth')
# Load the state dictionary into the model
clap_model.load_state_dict(state_dict)
model = clap_model

weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
ckpt = torch.load(weights_path, map_location=torch.device('cpu'))['model']
clap_model.load_state_dict(ckpt,strict=False)
model = clap_model

model_aud = model.audio_encoder
model_tex = model.caption_encoder

print('*'*20)
# Computing text embeddings

# Computing audio embeddings
y_preds, y_labels = [], []


for batch in tqdm(test_loader):
    audio_b, text_b = batch['audio'], batch['text']
    audio_b_f, audio_b_r,audio_b_w = audio_b
    text_b_f, text_b_r, text_b_w = text_b
    extracted_texts = [item[0] for item in text_b]

    print(extracted_texts)
    
    text_embeddings_f = clap_wrap.CLAPWrap(model).get_text_embeddings(extracted_texts)
    one_hot_target = text_b_f[0]
    print('one_hot_target',one_hot_target)

    # get the 1 hot class for this text
    one_hot_target = one_hot_encode(one_hot_target)
    one_hot_target = np.array([1,0,0]).reshape(1,-1)
    
    print(one_hot_target.shape)


    audio_embeddings = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_f)
    exit()
    #one_hot_target = clap_wrap.CLAPWrap(model).get_text_embeddings(one_hot_target)
    
    similarity = clap_wrap.CLAPWrap(model).compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target)


y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)

print(y_labels.shape)
print(y_preds.shape)


acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))

"""
The output:

ESC50 Accuracy: 93.9%

"""
