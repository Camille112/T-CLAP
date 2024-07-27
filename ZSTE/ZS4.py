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
from itertools import combinations, product
import warnings
import os
from clap_dds import AudioTextDataset
import clap_dds as dds
from msclap import CLAP

# Define helper functions and dataset configuration

def generate_combinations(word_list, base_words):
    """
    Generate all possible combinations of phrases using given words and base words.

    Args:
        word_list (list): List of words to choose from.
        base_words (list): List of base words to ensure presence in combinations.

    Returns:
        list: List of generated combinations in the format "<word> before <word>" and "<word> while <word>".
    """
    # Filter out words from word_list that are already in base_words
    filtered_word_list = [word for word in word_list if word not in base_words]

    # Select 5 words from the filtered word_list
    selected_words = combinations(filtered_word_list, 5)

    # Create a list to store all possible combinations
    all_combinations = []

    # Iterate through each combination of 5 words
    for words in selected_words:
        # Add the base words to the combination
        full_set = words + tuple(base_words)

        # Generate all possible pairings
        for word_i, word_j in product(full_set, repeat=2):
            if word_i != word_j:
                all_combinations.append(f"{word_i} before {word_j}")
                all_combinations.append(f"{word_i} while {word_j}")

    return all_combinations

def ensure_and_replace(word_list, must_have_words):
    """
    Ensure that certain must-have words are present in the word list, replacing random words if necessary.

    Args:
        word_list (list): List of words.
        must_have_words (list): List of words that must be included in the word list.

    Returns:
        list: Adjusted word list containing the must-have words.
    """
    # Check if the must-have words are already in the list
    missing_words = [word for word in must_have_words if word not in word_list]

    # If missing words are found, replace random elements in the list with them
    if missing_words:
        for word in missing_words:
            # Randomly choose an index to replace
            replace_index = random.randint(0, len(word_list) - 1)
            word_list[replace_index] = word

    return word_list

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

def torch_device_select(gpu):
    """
    Select the appropriate device (CPU or GPU) for PyTorch.

    Args:
        gpu (bool): Whether to use GPU if available.

    Returns:
        str: Device type ('cpu' or 'cuda').
    """
    # Check GPU availability & return device type
    if torch.cuda.is_available() and not gpu:
        warnings.warn("GPU is available but not used.")
        return 'cpu'
    elif not torch.cuda.is_available() and gpu:
        warnings.warn("GPU is not available but set to used. Using CPU.")
        return 'cpu'
    elif torch.cuda.is_available() and gpu:
        return 'cuda'
    else:
        return 'cpu'

def one_hot_vector(word, word_list):
    """
    Create a one-hot vector for a given word from a list of words.

    Args:
        word (str): The word to be one-hot encoded.
        word_list (list): List of all possible words.

    Returns:
        numpy.ndarray: One-hot encoded vector.
    """
    # Initialize a vector of zeros with length equal to the number of words in the list
    vector = [0] * len(word_list)

    # Find the index of the word in the word list and set that position to 1
    if word in word_list:
        index = word_list.index(word)
        vector[index] = 1

    return np.array(vector).reshape(1, -1)

# File paths and model configuration
filepath = '/Users/anshumansinha/Downloads/HW4_/esc50-temporal-pairs-reduced3.json'
file_path = '/Users/anshumansinha/Downloads/HW4_/words_without_underscore.txt'
root_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/audio/'
weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/model_epoch_10.pth'
# Other possible weights paths (commented out)
# weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
# weights_path = '/Users/anshumansinha/Downloads/HW4_/model_epoch_100_1.51.pth'
# weights_path = '/Users/anshumansinha/Downloads/HW4_/model_epoch_82_0.19.pth'

# Load the dataset
dataset = AudioTextDataset(json_file=filepath)
train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset)

# Create data loaders for the datasets
batch_size = 1
train_loader, val_loader, test_loader = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
save_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model/'

# Load and initialize CLAP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clap_model = CLAP(weights_path, version='2022', use_cuda=False)

# Compute audio embeddings and make predictions
y_preds, y_labels = [], []

# Reading the word list from a file
with open(file_path, 'r') as file:
    words_list = [line.strip() for line in file]

count = 0
for batch in tqdm(test_loader):
    count += 1
    audio_, text_ = batch['audio'], batch['text']

    audio_a, audio_b, audio_b_f, audio_b_r, audio_b_w = audio_
    text_a, text_b, text_b_f, text_b_r, text_b_w = text_
    one_hot_target = text_b_f[0]

    base_words = [text_a[0], text_b[0]]

    # Generate combinations and ensure must-have words are included
    combinations_l = generate_combinations(words_list, base_words)
    combinations_l = random.sample(combinations_l, 50)
    must_have_words = [text_b_f[0], text_b_w[0]]

    adjusted_list = ensure_and_replace(combinations_l, must_have_words)
    one_hot_target = one_hot_vector(one_hot_target, adjusted_list)

    prompt = 'this is a sound of '
    y = [prompt + x for x in adjusted_list]
    
    # Get text and audio embeddings using CLAP model
    text_embeddings_f = clap_model.get_text_embeddings(y)
    audio_embeddings = clap_model.get_audio_embeddings([audio_b_f[0]], resample=True)

    # Compute similarity between audio and text embeddings
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings_f)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()

    y_preds.append(y_pred)
    y_labels.append(one_hot_target)

# Print shapes of labels and predictions
print(y_labels[0].shape)
print(y_preds[0].shape)

# Concatenate all labels and predictions
y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)

print(y_labels.shape)
print(y_preds.shape)

# Calculate accuracy of the model
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))
