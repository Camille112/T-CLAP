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

# Load dataset
root_path = "root_path" # Folder with ESC-50-master/
dataset = ESC50(root=root_path, download=False) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]

#print(y)

text_d_list = [d['text_d'][0] for d in test_dataset] 
new_list = ['The sound of ' + element.split(' of ')[1] for element in text_d_list]
new_list = set(new_list)

print(new_list)
print(len(new_list))

# Load and initialize CLAP
path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
#path = '/Users/anshumansinha/Downloads/HW4_/HW5/CLAP_weights_2022.pth'

clap_model = CLAP(model_fp = path,version = '2022', use_cuda=False)
#text_list_emb = clap_model.get_text_embeddings(y)
text_list_emb = clap_model.get_text_embeddings(new_list)

count = 0
preds = []
y_preds, y_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader):
    #for i in tqdm(range(len(dataset))):
        
        '''
        x, _, one_hot_target = dataset.__getitem__(i)
        audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
        '''
        
        #ground_truth = [0 for _ in range(len(test_loader))]
        #ground_truth[count] = 1
        #ground_truth = torch.tensor([ground_truth])

        audio_, text_ = batch['audio_2'], batch['text_d']
        #audio_, text_ = batch['audio_1'], batch['text_s']

        # Extracting the last three words from the text
        text_last_three_words = ' '.join(text_[0][0].split()[3:])
        one_hot_vector = [1 if text_last_three_words in sound else 0 for sound in new_list]

        # text_s = 'The sound of' + text_[0][0].split('of')[1]

        # one_hot_target = find_ground_truth_label(y,text_[0][0])
        one_hot_target = torch.tensor(one_hot_vector)

        audio_embeddings = clap_model.get_audio_embeddings(audio_[0])

        similarity = clap_model.compute_similarity(audio_embeddings, text_list_emb)
        ranking = torch.argsort(similarity,descending= True)
        ind_x = torch.argmax(one_hot_target)

        '''
        x1 = 'this is the sound of ' + _
        one_hot_vector = [1 if sound == x1 else 0 for sound in y]
        '''

        pred = torch.where(ranking==ind_x.item())[1]
        pred = pred.cpu().numpy()
        preds.append(pred)

        #print('pred',pred)
        #print('pred_hello',np.argmax(pred))

        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.view(1,-1).detach().cpu().numpy())
        count+=1
        
        '''
        print('one_hot_target',one_hot_target)
        print('y_pred',y_pred)
        print(torch.argmax(one_hot_target))
        print(np.argmax(y_pred, axis=1))
        '''


for i, elem in enumerate(preds):
    # Convert to a NumPy array if it's not already one
    if not isinstance(elem, np.ndarray):
        elem = np.array(elem)
    # Print the index and shape of each element
    # print(f"Index: {i}, Shape: {elem.shape}")

preds = np.array(preds)

metrics = {}
metrics[f"mean_rank"] = preds.mean() + 1
metrics[f"median_rank"] = np.floor(np.median(preds)) + 1

for k in [1, 5, 10]:
    metrics[f"R@{k}"] = np.mean(preds < k)

# map@10
metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

print(
    f"Zeroshot Classification Results: "
    + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
)

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
print(y_labels.shape)
print(y_preds.shape)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))
'''
# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())


y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))
'''

"""
The output:

ESC50 Accuracy: 93.9%

"""