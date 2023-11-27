import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchaudio.transforms import Resample
import torchaudio

def read_csv(path):
    return pd.read_csv(path)

class AudioDataset(Dataset):
    def __init__(self, csv_df, audio_dir, transform=None, target_sample_rate=16000):
        self.labels = csv_df
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_name = self.labels.iloc[idx]['filename']
        label = self.labels.iloc[idx]['category']
        audio_path = os.path.join(self.audio_dir, audio_name)
        return audio_path, label

def split_dataset(df_in, test_size=0.2, random_state=42):
    labels = df_in
    train_df, val_df = train_test_split(labels, test_size=test_size, random_state=random_state)
    return train_df, val_df


def create_data_loaders(train_df, val_df, audio_dir, batch_size=32, num_workers=4):
    train_dataset = AudioDataset(csv_df=train_df, audio_dir=audio_dir)
    val_dataset = AudioDataset(csv_df=val_df, audio_dir=audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
