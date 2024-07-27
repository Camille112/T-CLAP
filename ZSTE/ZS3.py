"""
This version of the code is developed by anshs@gatech.edu
"""

import torch
import numpy as np
import random
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from clap_dds import AudioTextDataset
import clap_dds as dds
from msclap import CLAP
import os

# Load dataset configuration from a YAML file
def load_config(file_path):
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

# Set the top two maximum values in an array to one, rest to zero
def set_top_two_max_to_one(array):
    """
    Set the top two maximum values in an array to one, and the rest to zero.

    Args:
        array (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Array with the top two values set to one, others set to zero.
    """
    # Flatten the array to ensure it's one-dimensional
    flattened_array = array.flatten()

    # Find the indices of the two largest values
    indices_of_max_values = np.argpartition(flattened_array, -2)[-2:]

    # Create a new array of zeros with the same shape as the flattened array
    result_array = np.zeros_like(flattened_array)

    # Set the top two positions to 1
    result_array[indices_of_max_values] = 1

    return np.array(result_array)

# Load the dataset from a JSON file
filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
dataset = AudioTextDataset(json_file=filepath)

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset, test_size=0.2, random_state=42)

# Create data loaders for the datasets
batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

# Load the model configuration
config_path = '../configs/config_2022.yml'  # Replace with your actual file path
config = load_config(config_path)

# Load text data for audio-text pairing
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

prompt = 'this is a sound of '
y = [prompt + x for x in words_list]

# Load the CLAP model
weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
clap_model = CLAP(weights_path, version='2022', use_cuda=False)

# Print the list of audio-text pairs
print('*' * 20)
print('y', y)

# Get text embeddings using CLAP model
text_embeddings_f = clap_model.get_text_embeddings(y)
print(text_embeddings_f.shape)

# Initialize lists for predictions and labels
y_preds, y_labels = [], []

# Iterate through the test dataset
for batch in tqdm(test_loader):
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r, audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_

    text_b_t = [text_b_f[0], text_b_r[0], text_b_w[0]]

    # Define probabilities for selecting texts
    probabilities = [0.5, 0.5, 0.0]
    # Select one of the texts based on the given probabilities
    extracted_texts = random.choices(text_b_t, probabilities, k=1)[0]

    # One-hot encoding depending on the selected text
    if extracted_texts == text_b_f[0]:
        one_hot_target = np.array([1, 0, 0]).reshape(1, -1)
    elif extracted_texts == text_b_r[0]:
        one_hot_target = np.array([0, 1, 0]).reshape(1, -1)
    else:
        one_hot_target = np.array([0, 0, 1]).reshape(1, -1)

    # Select corresponding audio based on the extracted text
    if extracted_texts == text_b_f[0]:
        aud_s = [audio_b_f[0]]
    elif extracted_texts == text_b_r[0]:
        aud_s = [audio_b_r[0]]
    else:
        aud_s = [audio_b_w[0]]

    prompt = 'This is a sound of '
    y = [prompt + x for x in text_b_t]

    # Get text and audio embeddings using CLAP model
    text_embeddings_f = clap_model.get_text_embeddings(y)
    audio_embeddings = clap_model.get_audio_embeddings(aud_s, resample=True)

    # Compute similarity between audio and text embeddings
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    
    # Append predictions and labels for accuracy calculation
    y_preds.append(y_pred)
    y_labels.append(one_hot_target)

# Concatenate all labels and predictions
y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)

# Print shapes of labels and predictions
print(y_labels.shape)
print(y_preds.shape)

# Print predicted and actual labels
print('hello', np.argmax(y_preds, axis=1))
print('hello', np.argmax(y_labels, axis=1))

# Calculate accuracy of the model
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))
