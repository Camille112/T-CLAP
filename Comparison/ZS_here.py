"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

from msclap import CLAP
from esc_50_ds import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import glob
import json
import torch
import numpy as np
import os
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import dataset
import clap_dds_2 as dds
from clap import CLAP as CLAPS
from msclap import CLAP
import clap_wrap
from tqdm import tqdm

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

# Function to find the ground truth label
def find_ground_truth_label(labels, text):
    # Extract the key sound from the text
    key_sound = text.replace('Single sound of ', '').strip()

    # Initialize a list with zeros
    ground_truth = [0] * len(labels)

    # Find the index of the label matching the key sound and set it to 1
    for i, label in enumerate(labels):
        if key_sound in label:
            ground_truth[i] = 1
            break

    return ground_truth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1

# data specific
filepath = '/Users/anshumansinha/Downloads/HW3/esc50-temporal-pairs-reduced2_0.json'
#filepath = '/Users/anshumansinha/Downloads/HW3/inp.json'
dataset = dds.AudioTextDataset(json_file=filepath)
train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset)
# [{'audio_1': ['/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/5-9032-A-0.wav'], 'audio_2': ['TemporalAudio/9032-before-212054.wav'], 'text_s': ['Single sound of dog'], 'text_d': ['Combined sound of dog and vacuum cleaner']}]
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

# model specific
#weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/best_model_0/model_epoch_20.pth'
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)
# Load and initialize CLAP
weights_path = '/Users/anshumansinha/Downloads/HW4_/HW5/CLAP_weights_2022.pth'

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

# 'model' key from check-pont dic
ckpt = torch.load(weights_path, map_location=torch.device('cpu'))['model']
clap_model.load_state_dict(ckpt,strict=False)

# Load dataset
root_path = "root_path" # Folder with ESC-50-master/
dataset = ESC50(root=root_path, download=False) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]

# Computing text embeddings
text_embeddings = clap_wrap.CLAPWrap(clap_model).get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_wrap.CLAPWrap(clap_model).get_audio_embeddings([x], resample=True)
    similarity = clap_wrap.CLAPWrap(clap_model).compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())
    
    print(one_hot_target)
    exit()


y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))

"""
The output:

ESC50 Accuracy: 93.9%

"""