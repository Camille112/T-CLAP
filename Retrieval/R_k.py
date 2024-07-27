"""
This script performs zero-shot retrieval using different models on audio-text datasets.
It allows for flexible configuration of models, top-k retrieval, and datasets.

Classes:
    - AudioTextDataset: Dataset class for loading audio-text pairs.
    - CLAPModelWrapper: Wrapper for CLAP model to handle audio and text embeddings.

Functions:
    - load_model: Loads the selected model with given weights.
    - get_embeddings: Computes embeddings for audio and text data.
    - calculate_top_k_accuracy: Calculates the top-k accuracy for the retrieval task.
    - main: The main function to run the retrieval and evaluation.

Usage:
    1. Set the desired model, top-k values, and dataset path.
    2. Run the script to obtain the retrieval performance.
"""

from msclap import CLAP
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from clap import CLAP as CLAPS
from clap_dds import AudioTextDataset
import clap_ds as ds
import clap_dds as dds

class AudioTextDataset(Dataset):
    def __init__(self, json_file):
        # Load data from JSON file
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def load_model(model_name, weights_path, use_cuda=False):
    if model_name == 'CLAP':
        model = CLAP(weights_path, version='2022', use_cuda=use_cuda)
    # Add other model options here
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model

def get_embeddings(model, list_texts, list_audio):
    text_embeddings = model.get_text_embeddings(list_texts)
    audio_embeddings = model.get_audio_embeddings(list_audio, resample=False)
    return text_embeddings, audio_embeddings

def calculate_top_k_accuracy(model, test_loader, top_k=5):
    score = 0
    total_items = 0
    list_texts = []
    list_audio = []
    
    # Precompute embeddings for all texts and audios in the test set
    for batch in test_loader:
        audio_, text_ = batch['audio'], batch['text']
        audio_a = audio_[0]
        text_a = text_[0]
        list_texts.append(text_a)
        list_audio.append(audio_a)

    classes = list_texts
    text_embeddings = model.get_text_embeddings(list_texts)
    audio_embeddings = model.get_audio_embeddings(list_audio, resample=False)

    for batch in tqdm(test_loader):
        audio_, text_ = batch['audio'], batch['text']
        audio_a, text_a = audio_[0], text_[0]
        
        ground_truth_text = text_a
        ground_truth_audio = audio_a

        text_embeddings = model.get_text_embeddings([ground_truth_text])
        audio_embeddings = model.get_audio_embeddings([ground_truth_audio], resample=False)

        similarity = model.compute_similarity(audio_embeddings, text_embeddings)
        similarity = F.softmax(similarity, dim=1)
        values, indices = similarity[0].topk(top_k)

        top_k_texts = [classes[index] for index in indices]

        if ground_truth_text in top_k_texts:
            score += 1

        total_items += 1

    return score / total_items


if __name__ == "__main__":
  
    filepath = '/path/to/your/dataset.json'  # Set your dataset path here
    model_name = 'CLAP'
    weights_path = '/path/to/your/weights.pth'
    top_k = 5  # Set top-k retrieval value

    dataset = AudioTextDataset(json_file=filepath)
    train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(model_name, weights_path, use_cuda=False)
    final_score = calculate_top_k_accuracy(model, test_loader, top_k=top_k)

    print("Final Score:", final_score)
