import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
from clap import CLAP as CLAPS

from msclap import CLAP

import yaml

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm
import pandas as pd
import os
import torch.nn as nn
import torch
from datasets import load_dataset


class MultiModalAttention(nn.Module):
    def __init__(self, audio_embed_dim, text_embed_dim, attention_dim):
        super(MultiModalAttention, self).__init__()
        self.audio_attention = nn.Linear(audio_embed_dim, attention_dim)
        self.text_attention = nn.Linear(text_embed_dim, attention_dim)
        self.combined_attention = nn.Linear(attention_dim, 1)

    def forward(self, audio_embed, text_embed):
        audio_attn = self.audio_attention(audio_embed)
        text_attn = self.text_attention(text_embed)
        
        # Combine the attention weights
        combined = torch.tanh(audio_attn + text_attn)
        attention_weights = F.softmax(self.combined_attention(combined), dim=1)

        return attention_weights

class ExtendedModel(nn.Module):
    def __init__(self, base_model):
        super(ExtendedModel, self).__init__()

        self.audio_encoder = base_model.audio_encoder
        self.text_encoder = base_model.caption_encoder

        text_embed_dim = self.text_encoder.projection.linear2.out_features
        audio_embed_dim = self.audio_encoder.projection.linear2.out_features
        
        # Attention mechanism
        attention_dim = 128  
        self.attention = MultiModalAttention(audio_embed_dim, text_embed_dim, attention_dim)

        # Deep transformation network
        hidden_dim = 512 
        self.transformation = nn.Sequential(
            nn.Linear(audio_embed_dim + text_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)  # regularization
        )

        # Task-specific layer: Adjust depending on your task
        self.task_layer = nn.Linear(hidden_dim, 5)  # Assuming a task-specific size

    def forward(self, audio, text):
        audio_embed = self.audio_encoder(audio)
        text_embed = self.text_encoder(text)

        attention_weights = self.attention(audio_embed, text_embed)

        # Weighted combination of features
        combined = attention_weights * audio_embed + attention_weights * text_embed
        # Deep transformation
        transformed = self.transformation(combined)
        # Task-specific layer
        output = self.task_layer(transformed)

        return output