
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
#import positional_encoder as pe
import time
import clap_ds as ds
import clap_wrap

from cont_loss import (
    LossAddition,
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
batch_size = 2

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
        
        batch_size = 32
        sample_weights = torch.ones(2 * batch_size, device=self.device)
        sample_weights[batch_size // 2:] *= self.beta
        
        # define the losses
        self.contrastive_lambda = self.CONFIG['contrastive_lambda']

        alpha_matrix = self.alpha_cross * np.ones((batch_size, batch_size))
        alpha_matrix[np.arange(batch_size), np.arange(batch_size)] = self.alpha_same
        self.contrastive = LossAddition(
            [
                T2AContraLoss(sample_weights, alpha_matrix),
                A2TContraLoss(sample_weights, alpha_matrix),
            ],
        )
    

    def compute_batch_losses(self, outputs): # outputs = embedding coming from model.
    
        audio_forward = outputs["audio_embeddings_f"]
        audio_reverse = outputs["audio_embeddings_r"]
        audio = torch.cat([audio_forward, audio_reverse], dim=0)
    
        text_forward = outputs["text_embeddings_f"]
        text_reverse = outputs["text_embeddings_r"]
        text = torch.cat([text_forward, text_reverse], dim=0)

        # contrastive loss
        loss_contrastive = self.contrastive(batch_audio=audio, batch_text=text)

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

        for batch in training_loader:

            audio_b, text_b = batch
            #audio_b_f, audio_b_r, text_b_f, text_b_r = batch

            text_embeddings= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b)
            audio_embeddings = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b)# torch.Size([32, 1024])

            text_embeddings_f= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b)
            text_embeddings_r= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b)
            audio_embeddings_f = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b)
            audio_embeddings_r = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b)

            '''
            text_embeddings_f= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b_f)
            text_embeddings_r= clap_wrap.CLAPWrap(model).get_text_embeddings(text_b_r)
            audio_embeddings_f = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_f)
            audio_embeddings_r = clap_wrap.CLAPWrap(model).get_audio_embeddings(audio_b_r)
            '''

            input_dic = {
                'text_embeddings_f': text_embeddings_f,  # Placeholder for forward text embeddings
                'text_embeddings_r': text_embeddings_r,  # Placeholder for reverse text embeddings
                'audio_embeddings_f': audio_embeddings_f,  # Placeholder for forward audio embeddings
                'audio_embeddings_r': audio_embeddings_f,  # Placeholder for reverse audio embeddings
            }

            loss = self.compute_batch_losses(input_dic) # input_list : []

            print(loss)
            exit()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
        
        return total_loss

    def evaluate(model, eval_data: Tensor) -> float:
        
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

    def evaluate_test(model) -> float:
        
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

    # Load your configuration file
    config_path = '../configs/config_2022.yml'  # Replace with your actual file path
    config = load_config(config_path)

    # Load and initialize CLAP
    weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
    c_model = CLAP(weights_path, version = '2022', use_cuda=False)

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
    
    # model = ExtendedModel(clap_model)
    model = c_model.load_clap()[0]
    model = clap_model

    model_aud = model.audio_encoder
    model_tex = model.caption_encoder

    for name, param in model_tex.named_parameters():
        # Check if 'base.pooler' or 'projection' is in the parameter name
        if "base.pooler" not in name and "projection" not in name and "base.encoder.layer.11" not in name:
            param.requires_grad = False  # Freeze this parameter
        else:
            # Otherwise, don't freeze it. It will be updated during training.
            param.requires_grad = True

    trainable_params1 = sum(p.numel() for p in model_tex.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params1}")

    trainable_params2 = sum(p.numel() for p in model_tex.parameters())
    print(f"Total number of trainable parameters: {trainable_params2}")

    print(f'Percentage of trainable parameters Text: {(trainable_params1/trainable_params2)*100}%')
    
    print('*'*20)

    for name, param in model_aud.named_parameters():
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
    print(f"Total number of trainable parameters: {trainable_params2}")

    print(f'Percentage of trainable parameters Audio: {(trainable_params1/trainable_params2)*100}%')

    loss_function = torch.nn.MSELoss()

    lr = 5.0  # learning rate , too high -> check

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Now, only the parameters of the last layer will be updated.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float('inf')
    epochs = 1

    audio_dir = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/data_combo/combo_audio/'
    csv_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP-main/examples/root_path/ESC-50-master/data_combo/combined.csv'

    # Split the dataset into training and validation

    read_cs = pd.read_csv(csv_path)
    train_df, val_df = ds.split_dataset(read_cs)

    # Create DataLoaders for training and validation sets
    train_loader, val_loader = ds.create_data_loaders(train_df, val_df, audio_dir)

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