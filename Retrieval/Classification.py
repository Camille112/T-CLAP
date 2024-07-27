"""
This version of the code is developed by anshs@gatech.edu.

This script performs audio classification using different models on audio-text datasets.
It supports flexible configuration for different models, dataset paths, and classification tasks.

Classes:
    - AudioTextDataset: Dataset class for loading audio-text pairs.

Functions:
    - load_config: Loads configuration from a YAML file.
    - torch_device_select: Selects the appropriate device (CPU or GPU) for computation.
    - load_model: Loads and initializes the specified model with weights.
    - one_hot_encode: One-hot encodes the first word in the given text based on a list of words.
    - main: Main function to load data, perform classification, and compute accuracy.

Usage:
    1. Set the desired model, dataset path, and other configurations.
    2. Run the script to obtain classification performance.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import yaml
import warnings
import os

from clap import CLAP as CLAPS
from msclap import CLAP
from clap_dds_1 import AudioTextDataset
import clap_wrap

def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def torch_device_select(gpu):
    """Selects the device type (CPU or GPU) based on availability and user preference."""
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

def load_model(model_name, config, weights_path):
    """Load and initialize the specified model with weights."""
    if model_name == 'CLAP':
        model = CLAPS(
            audioenc_name=config['audioenc_name'],
            sample_rate=config['sampling_rate'],
            window_size=config['window_size'],
            hop_size=config['hop_size'],
            mel_bins=config['mel_bins'],
            fmin=config['fmin'],
            fmax=config['fmax'],
            classes_num=config['num_classes'],
            out_emb=config['out_emb'],
            text_model=config['text_model'],
            transformer_embed_dim=config['transformer_embed_dim'],
            d_proj=config['d_proj']
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))['model']
    model.load_state_dict(state_dict, strict=False)
    return model

def one_hot_encode(text, words_list):
    """One-hot encode the first word in the given text."""
    first_word = text.split()[0]
    one_hot_vector = np.zeros((1, len(words_list)))

    if first_word in words_list:
        index = words_list.index(first_word)
        one_hot_vector[0, index] = 1

    return one_hot_vector

if __name__ == "__main__":
    
  filepath = '/path/to/your/dataset.json'  # Set your dataset path here
    model_name = 'CLAP'
    weights_path = '/path/to/your/weights.pth'
    config_path = '/path/to/your/config.yml'  # Replace with your actual file path
    words_list_path = '/path/to/words_without_underscore.txt'  # Replace with your actual file path

    # Load configuration
    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration.")
        return

    # Load dataset
    dataset = AudioTextDataset(json_file=filepath)
    train_dataset, val_dataset, test_dataset = ds.split_dataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load words list
    with open(words_list_path, 'r') as file:
        words_list = [line.strip() for line in file]

    # Load model
    model = load_model(model_name, config, weights_path)
    model.eval()

    y_preds, y_labels = [], []

    for batch in tqdm(test_loader):
        audio_b, text_b = batch['audio'], batch['text']
        audio_b_f = audio_b[0]
        extracted_texts = [item[0] for item in text_b]

        text_embeddings_f = clap_wrap.CLAPWrap(model).get_text_embeddings(extracted_texts)
        one_hot_target = one_hot_encode(extracted_texts[0], words_list)

        audio_embeddings = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_f)

        similarity = clap_wrap.CLAPWrap(model).compute_similarity(audio_embeddings, text_embeddings_f)
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target)

    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    print('Classification Accuracy:', acc)
