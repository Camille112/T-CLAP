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

        s_path = './TemporalAudio_2/'

        text_a = item["A"][0]
        path_a = item["A"][1]
        texts.append(text_a)
        audio_paths.append(s_path+path_a)

        text_b = item["B"][0]
        path_b = item["B"][1]
        texts.append(text_b)
        audio_paths.append(s_path+path_b)

        text_a_before_b = item["A before B"][0]
        path_a_before_b = item["A before B"][1]
        texts.append(text_a_before_b)
        audio_paths.append(path_a_before_b)

        text_b_before_a = item["B before A"][0]
        path_b_before_a = item["B before A"][1]
        texts.append(text_b_before_a)
        audio_paths.append(path_b_before_a)

        text_a_while_b = item["A while B"][0]
        path_a_while_b = item["A while B"][1]
        texts.append(text_a_while_b)
        audio_paths.append(path_a_while_b)
        
        sample = {'audio': audio_paths, 'text': texts}  
        if self.transform:
            sample = self.transform(sample)

        return sample


def split_dataset(dataset, test_size=0.2, random_state=42):
    train, test = train_test_split(dataset, test_size=0.1,random_state=random_state)
    train, val = train_test_split(train, test_size=test_size,random_state=random_state)
    return train, val, test

def create_data_loaders(train_data, val_data, test_data, batch_size=3):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
