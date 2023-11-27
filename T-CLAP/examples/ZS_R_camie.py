"""
This is an example using CLAP for zero-shot inference.
@ effects.
"""
from msclap import CLAP
import torch.nn.functional as F
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


filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset,test_dataset = dds.split_dataset(dataset)

batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)


weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
#weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
clap_model = CLAP(weights_path, version = '2022', use_cuda=False)
# Load the state dictionary into the model

list_texts = []
list_audio = []
for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r, audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_
    list_texts.append(text_a[0])
    list_audio.append(audio_a[0])

classes = list_texts
# compute text embeddings from natural text
text_embedding = clap_model.get_text_embeddings(list_texts)
audio_embedding = clap_model.get_audio_embeddings(list_audio, resample=False)

score = 0
total_items = 0

for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r, audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_
    
    ground_truth_text = text_a[0]
    ground_truth_audio = audio_a[0]

    text_embeddings = clap_model.get_text_embeddings(ground_truth_text)
    audio_embeddings = clap_model.get_audio_embeddings([ground_truth_audio], resample=False)


    # Compute the similarity between audio_embeddings and text_embeddings
    similarity = clap_model.compute_similarity(audio_embeddings, text_embedding)
    similarity = F.softmax(similarity, dim=1)
    values, indices = similarity[0].topk(10)

    # Retrieve texts corresponding to the top 5 indices
    top_5_texts = [classes[index] for index in indices]

    # Check if ground_truth is among the top 5 texts
    if ground_truth in top_5_texts:
        score += 1

    total_items += 1

# Calculate final score
final_score = score / total_items
print("Final Score:", final_score)

"""
The output (the exact numbers may vary):

Ground Truth: coughing
Top predictions:

        coughing: 98.55%
        sneezing: 1.24%
drinking sipping: 0.15%
       breathing: 0.02%
  brushing teeth: 0.01%
"""
