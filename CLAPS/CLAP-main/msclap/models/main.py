
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils
import data_ssl as ds
import matplotlib.pyplot as plt
from SSL_model import ExtendedModel

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

import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import positional_encoder as pe
import time


def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

# Hyperparams
test_size = 0.1
batch_size = 2

class Trainer():

    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG
        self.device = torch_device_select(self.CONFIG['gpu'])
            
    def train(model: nn.Module) -> None:
        
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        print('epochs : ', epoch)

        dataset = load_dataset("MLCommons/peoples_speech", split='train', streaming=True)
        dataset_head = dataset.take(2)
        # dataset_head = (list(dataset_head))
        training_loader = torch.utils.data.DataLoader(dataset_head, batch_size=2, shuffle=True)

        for batch, i in training_loader:

            data, targets = batch, i # torch.Size([2, 500, 12])
            output = model(data) # torch.Size([2, 500, 10])
            audio_path = i['audio']['path']  # This is your audio data
            text = i['text']  # This is your sampling rate

            audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
            text_embeddings = clap_model.get_audio_embeddings(audio_embeddings, text_embeddings)

            loss = loss_function(output, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
        
        return total_loss

    def evaluate(model: nn.Module, eval_data: Tensor) -> float:
        
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        batch_l = len(eval_data)
        with torch.no_grad():
            for batch,i in validation_loader:
                data, targets = batch, i
                output = model(data)
                targets = targets.reshape(-1, 75)
                total_loss += loss_function(output, targets).item()
        return total_loss/batch_l

    def evaluate_test(model: nn.Module) -> float:
        
        model.eval()  # turn on evaluation mode
        total_loss = 0.

        for i, v in training_loader: # replace with test_loader.
            x_dat, tar = i,v
            break
        
        #print('x_data',x_dat.shape) # ([500, 12])
        output = model(x_dat)

        output = output[0]
        tar = tar.reshape(-1, 75)
        target = tar[0]

        total_loss = loss_function(output, target).item()
        return total_loss


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2

    clap_model = CLAPS(
                audioenc_name=config['audioenc_name'],
                sample_rate=config['sampling_rate'],  # 'sample_rate' seems to be the correct parameter name
                window_size=config['window_size'],
                hop_size=config['hop_size'],
                mel_bins=config['mel_bins'],
                fmin=config['fmin'],
                fmax=config['fmax'],
                classes_num=config['num_classes'],  # It seems 'classes_num' is the correct parameter name
                out_emb=config['out_emb'],
                text_model=config['text_model'],
                transformer_embed_dim=config['transformer_embed_dim'],
                d_proj=config['d_proj']
            )

    # 'model' key from check-pont dic
    ckpt = torch.load(weights_path, map_location=torch.device('cpu'))['model']
    clap_model.load_state_dict(ckpt,strict=False)
    model = ExtendedModel(clap_model)

    loss_function = torch.nn.MSELoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = 1

    with TemporaryDirectory() as tempdir:
        
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        trainer = Trainer(CONFIG)

        for epoch in range(1, epochs + 1):
            
            epoch_start_time = time.time()
            print('trainer is running wild')

            tr_loss = trainer.train(model)
            val_loss = trainer.evaluate(model, val_data)

            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'valid loss {val_loss:5.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()

        model.load_state_dict(torch.load(best_model_params_path)) # load best model states
        test_loss = trainer.evaluate_test(model)

        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f}')
        print('=' * 89)


'''

if __name__ == "__main__":
    # Load your configuration file
    config_path = '../configs/config_2022.yml'  # Replace with your actual file path
    config = load_config(config_path)

    # Load and initialize CLAP
    weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
    c_model = CLAP(weights_path, version = '2022', use_cuda=False)

'''