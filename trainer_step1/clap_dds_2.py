import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class AudioTextDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_paths_1 = []
        audio_paths_2 = []

        texts_s = []
        texts_d = []

        path_aud_esc = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/'

        for i in item:

            audio_path_1 = i['path-a']
            audio_path_2 = i['path']
            text = i['text']

            words = text.split(" before ")
            text_b_single = 'Single sound of ' + words[0]
            text_b_double = 'Combined sound of ' + words[0] + ' and ' + words[1]

            audio_paths_1.append(path_aud_esc+audio_path_1)
            audio_paths_2.append(audio_path_2)
            texts_s.append(text_b_single)
            texts_d.append(text_b_double)
            break
        
        sample = {'audio_1': audio_paths_1,'audio_2': audio_paths_2, 'text_s': texts_s, 'text_d': texts_d}

        if self.transform:
            sample = self.transform(sample)

        return sample


def split_dataset(dataset, test_size=0.2):
    train, test = train_test_split(dataset, test_size=0.1)
    train, val = train_test_split(train, test_size=test_size)
    return train, val, test

def create_data_loaders(train_data, val_data, test_data, batch_size=3):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
