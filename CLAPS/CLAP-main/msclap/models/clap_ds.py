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
        
        '''
        
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample the audio if required
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Apply any additional transformations
        if self.transform:
            waveform = self.transform(waveform)
        '''

        return audio_path, label

def split_dataset(df_in, test_size=0.2, random_state=42):
    labels = df_in
    train_df, val_df = train_test_split(labels, test_size=test_size, random_state=random_state)
    return train_df, val_df


def create_data_loaders(train_df, val_df, audio_dir, batch_size=32, num_workers=4):
    train_dataset = AudioDataset(csv_df=train_df, audio_dir=audio_dir)
    val_dataset = AudioDataset(csv_df=val_df, audio_dir=audio_dir)

    # Print the first few items in the dataset
    '''
    for i in range(min(len(train_dataset), 10)):  # Print the first 10 or the total number of items
        waveform, label = train_dataset[i]
        print(f"Waveform: {waveform.shape} - Label: {label}")
    '''

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
