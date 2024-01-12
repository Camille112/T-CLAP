"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

#import sys
#sys.path.append('/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP/laion_clap')
import laion_clap
import TextEncoder as text_enc
import AudioEncoder as aud_enc 

#from msclap import CLAP
from esc_50_ds import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import torchaudio
import torchaudio.transforms as T

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

def default_collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        default_collate_err_msg_format.format(elem.dtype))

                return default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

def preprocess_audio(audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = load_audio_into_tensor(
                audio_file, 20, resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return default_collate(audio_tensors)

def read_audio(audio_path, resample=True):
        r"""Loads audio file or array and returns a torch tensor"""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        
        resample_rate = 32000
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        return audio_time_series, resample_rate

def load_audio_into_tensor(audio_path, audio_duration, resample=False):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = read_audio(audio_path, resample=resample)
        audio_time_series = audio_time_series.reshape(-1)

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

def _get_audio_embeddings( preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2])
            #Append [0] the audio emebdding, [1] has output class probabilities
            return audenc(preprocessed_audio)[0]

def get_audio_embeddings(audio_files, resample=True):
        r"""Load list of audio files and return a audio embeddings"""
        preprocessed_audio = preprocess_audio(audio_files, resample)
        return _get_audio_embeddings(preprocessed_audio)

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

# Load and initialize CLAP
path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/audio-text_retrieval/pretrained_models/ResNet38.pth'

texenc = text_enc.BertEncoder()
audenc = aud_enc.ResNet38()#, amodel= 'HTSAT-base')
# Load the pre-trained weights
state_dict = torch.load(path, map_location=torch.device('cpu'))
audenc.load_state_dict(state_dict,strict=False)

text_embed = texenc(y)
print(text_embed.shape) # (417, 512)

count = 0
preds = []
y_preds, y_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader):
    #for i in tqdm(range(len(dataset))):
        
        #x, _, one_hot_target = dataset.__getitem__(i)
        #audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
        
        #ground_truth = [0 for _ in range(len(test_loader))]
        #ground_truth[count] = 1
        #ground_truth = torch.tensor([ground_truth])

        audio_, text_ = batch['audio_2'], batch['text_d']
        audio_, text_ = batch['audio_1'], batch['text_s']

        # Extracting the last three words from the text
        text_last_three_words = ' '.join(text_[0][0].split()[3:])
        one_hot_vector = [1 if text_last_three_words in sound else 0 for sound in y]

        text_s = 'The sound of' + text_[0][0].split('of')[1]

        #one_hot_target = find_ground_truth_label(y,text_[0][0])
        one_hot_target = torch.tensor(one_hot_vector)

        x = audio_[0][0]

        # audio_embeddings = clap_model.get_audio_embeddings(audio_[0])
        aud_l = get_audio_embeddings(audio_files=[x], resample=True)
        aud_l = aud_l.view(-1,1)
        print(aud_l.shape)
        audio_embed = audenc(aud_l)

        similarity = torch.tensor(audio_embed) @ torch.tensor(text_embed).t()
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