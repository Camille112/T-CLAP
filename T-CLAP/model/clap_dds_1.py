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
        audio_paths = []
        texts = []
        for i in item:
            audio_path = i['path']
            text = i['text']
            audio_paths.append(audio_path)
            texts.append(text)
        
        sample = {'audio': audio_paths, 'text': texts}  
        if self.transform:
            sample = self.transform(sample)

        return sample


def split_dataset(dataset, test_size=0.2):
    train, test = train_test_split(dataset, test_size=0.1)
    train, val = train_test_split(train, test_size=test_size)
    return train, val, test

def create_data_loaders(train_data, val_data, test_data, batch_size=3):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
