
import utils
from torch.utils.data import DataLoader
import torch
import datetime
import numpy as np
import utils

from SSL_model import ExtendedModel

import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
from clap import CLAP as CLAPS

from msclap import CLAP
import warnings

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

import time

from clap_dds_2 import AudioTextDataset
import clap_dds_2 as dds

import clap_wrap

from cont_loss_0 import (
    Loss_total,
    T2AContraLoss,
    A2TContraLoss,
)

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None

def torch_device_select(gpu):
    # check GPU availability & return device type
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

# Hyperparams
test_size = 0.1
batch_size = 32

CONFIG = {}
CONFIG['gpu'] = True
CONFIG['contrastive_lambda'] = 0.8

class Trainer():

    def __init__(self, CONFIG) -> None:

        self.CONFIG = CONFIG
        self.device = torch_device_select(self.CONFIG['gpu'])
        self.alpha_same = 1.0
        self.alpha_cross = 1.0
        self.beta = 1.0
        self.contrastive_lambda = self.CONFIG['contrastive_lambda']

        alpha_matrix = self.alpha_cross * np.ones((batch_size, batch_size))
        alpha_matrix[np.arange(batch_size), np.arange(batch_size)] = self.alpha_same

        sample_weights = torch.ones(2 * batch_size, device=self.device)
        sample_weights[batch_size // 2:] *= self.beta
        
        self.contrastive = Loss_total(
            [
                T2AContraLoss(sample_weights, alpha_matrix),
                A2TContraLoss(sample_weights, alpha_matrix),
            ],
        )
    
    def compute_batch_losses(self, outputs): # outputs = embedding coming from model.

        audio_s = outputs["audio_embeddings_single"]
        audio_d = outputs["audio_embeddings_double"]

        audio = torch.cat([audio_s, audio_d], dim=0)
    
        text_s = outputs["text_embeddings_single"]
        text_d = outputs["text_embeddings_double"]

        text = torch.cat([text_s,text_d], dim=0)

        print('text',text.shape)
        print('audio',audio.shape)

        # contrastive loss
        loss_contrastive = self.contrastive(batch_audio=audio, batch_text=text)
        print('loss_contrastive',loss_contrastive)

        # total loss
        total_loss = self.contrastive_lambda * loss_contrastive

        return {"total": total_loss, "contrastive": loss_contrastive}
            
    def train(self,model) -> None:
        
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        print('epochs : ', epoch)
        training_loader = train_loader
        batch_l = len(training_loader)
        print(batch_l)

        for batch in training_loader:

            audio_b_single,audio_b_double, text_b_single, text_b_double = batch['audio_1'][0], batch['audio_2'][0], batch['text_s'][0], batch['text_d'][0]

            text_embeddings_single= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b_single)
            text_embeddings_double= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b_double)
            audio_embeddings_single = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_single)
            audio_embeddings_double = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_double)

            input_dic = {
                'text_embeddings_single': text_embeddings_single,  # Placeholder for forward text embeddings
                'text_embeddings_double': text_embeddings_double,  # Placeholder for reverse text embeddings
                'audio_embeddings_single': audio_embeddings_single,  # Placeholder for forward audio embeddings
                'audio_embeddings_double': audio_embeddings_double,  # Placeholder for reverse audio embeddings
                }

            #try:
            loss = self.compute_batch_losses(input_dic) # input_list : []
            loss = loss["total"]

            optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            #except:
            #    print('batch_fault')
            
            
        return total_loss/batch_l

    def evaluate(self,model) -> float:
        
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        batch_l = len(val_loader)

        with torch.no_grad():

            for batch in val_loader:
                
                audio_b_single,audio_b_double, text_b_single, text_b_double = batch['audio_1'], batch['audio_2'], batch['text_s'], batch['text_d']

                text_embeddings_single= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b_single)
                text_embeddings_double= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b_double)
                audio_embeddings_single = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_single)
                audio_embeddings_double = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_double)
            
                input_dic = {
                    'text_embeddings_single': text_embeddings_single,  # Placeholder for forward text embeddings
                    'text_embeddings_double': text_embeddings_double,  # Placeholder for reverse text embeddings
                    'audio_embeddings_single': audio_embeddings_single,  # Placeholder for forward audio embeddings
                    'audio_embeddings_double': audio_embeddings_double,  # Placeholder for reverse audio embeddings
                    }
                    

                try:
                    loss = self.compute_batch_losses(input_dic) # input_list : []
                    loss = loss["total"]
                    total_loss += loss.item()
                except:
                    print('batch_fault')

            return total_loss/batch_l

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    # Load your configuration file
    config_path = '../configs/config_2022.yml'  # Replace with your actual file path
    config = load_config(config_path)

    # Load and initialize CLAP
    weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'

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
                d_proj=config['d_proj'])

    # 'model' key from check-pont dic
    ckpt = torch.load(weights_path, map_location=torch.device('cpu'))['model']
    clap_model.load_state_dict(ckpt,strict=False)
    model = clap_model

    model_aud = model.audio_encoder
    model_tex = model.caption_encoder

    for name, param in model_tex.named_parameters():
        print(name)
        # Check if 'base.pooler' or 'projection' is in the parameter name
        if "base.pooler" not in name and "projection" not in name and "base.encoder.layer.11" not in name:
            param.requires_grad = False  # Freeze this parameter
        else:
            # Otherwise, don't freeze it. It will be updated during training.
            param.requires_grad = True

    trainable_params1 = sum(p.numel() for p in model_tex.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params1}")
    trainable_params2 = sum(p.numel() for p in model_tex.parameters())
    print(f"Total number of parameters: {trainable_params2}")
    print(f'Percentage of trainable parameters Text: {(trainable_params1/trainable_params2)*100}%')
    
    print('*'*20)

    for name, param in model_aud.named_parameters():
        print(name)
        # Check if 'base.pooler' or 'projection' is in the parameter name
        if "base.fc1" not in name and "base.fc_audioset" not in name and "projection" not in name:
            param.requires_grad = False  # Freeze this parameter
        else:
            # Otherwise, don't freeze it. It will be updated during training.
            param.requires_grad = True

    # After updating requires_grad, check which parameters are trainable
    for name, param in model_tex.named_parameters():
        print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")
    for name, param in model_aud.named_parameters():
        print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")


    trainable_params1 = sum(p.numel() for p in model_aud.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params1}")
    trainable_params2 = sum(p.numel() for p in model_aud.parameters())
    print(f"Total number of parameters: {trainable_params2}")

    print(f'Percentage of trainable parameters Audio: {(trainable_params1/trainable_params2)*100}%')

    loss_function = torch.nn.MSELoss()

    lr = 0.001  # learning rate , too high -> check

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Now, only the parameters of the last layer will be updated.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = 20

    filepath = '/Users/anshumansinha/Downloads/HW3/esc50-temporal-pairs-reduced2_0.json'

    dataset = AudioTextDataset(json_file=filepath)
    train_dataset, val_dataset, test_dataset = dds.split_dataset(dataset)
    batch_size = 4
    train_loader, val_loader, _ = dds.create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)
    save_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/msclap/models/best_model_0/'

    with TemporaryDirectory() as tempdir:
        
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        trainer = Trainer(CONFIG)

        for epoch in range(1, epochs + 1):
            
            epoch_start_time = time.time()
            print('trainer is running wild')

            tr_loss = trainer.train(model)
            val_loss = trainer.evaluate(model)

            elapsed = time.time() - epoch_start_time
            
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ' f'valid loss {val_loss:5.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)
            
            epoch_save_path = os.path.join(save_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), epoch_save_path)

            scheduler.step()

        model.load_state_dict(torch.load(best_model_params_path)) # load best model states
        #test_loss = trainer.evaluate_test(model)

        print('=' * 89)
        #print(f'| End of training | test loss {test_loss:5.2f}')
        print('=' * 89)