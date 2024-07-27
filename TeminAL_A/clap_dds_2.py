import torch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split

class AudioTextDataset(Dataset):
    """
    A custom Dataset class for loading audio and text data from a JSON file.

    Attributes:
        data (list): List of data items loaded from the JSON file.
        transform (callable, optional): Optional transform to be applied on a sample.

    Args:
        json_file (str): Path to the JSON file containing the data.
        transform (callable, optional): A function/transform to apply on the sample.
    """
    def __init__(self, json_file, transform=None):
        # Load data from the JSON file
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.transform = transform

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.

        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            dict: A dictionary containing audio paths and transformed text.
        """
        item = self.data[idx]
        audio_paths_1 = []
        audio_paths_2 = []
        texts_s = []
        texts_d = []

        # Base path for audio files
        path_aud_esc = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/'

        for i in item:
            audio_path_1 = i['path-a']
            audio_path_2 = i['path']
            text = i['text']

            # Split the text to create single and combined sound descriptions
            words = text.split(" before ")
            text_b_single = 'Single sound of ' + words[0]
            text_b_double = 'Combined sound of ' + words[0] + ' and ' + words[1]

            # Append paths and texts to the respective lists
            audio_paths_1.append(path_aud_esc + audio_path_1)
            audio_paths_2.append(audio_path_2)
            texts_s.append(text_b_single)
            texts_d.append(text_b_double)
            break  # Exiting after the first iteration for each item (appears intentional)

        sample = {'audio_1': audio_paths_1, 'audio_2': audio_paths_2, 'text_s': texts_s, 'text_d': texts_d}

        # Apply the transform if it exists
        if self.transform:
            sample = self.transform(sample)

        return sample

def split_dataset(dataset, test_size=0.2):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Three datasets (train, val, test) after splitting.
    """
    train, test = train_test_split(dataset, test_size=0.1)  # Small test set
    train, val = train_test_split(train, test_size=test_size)  # Split the rest into train and val
    return train, val, test

def create_data_loaders(train_data, val_data, test_data, batch_size=3):
    """
    Creates DataLoader objects for training, validation, and test datasets.

    Args:
        train_data (Dataset): Training dataset.
        val_data (Dataset): Validation dataset.
        test_data (Dataset): Test dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: DataLoader objects for train, val, and test datasets.
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
