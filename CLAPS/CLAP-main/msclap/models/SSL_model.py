import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
from clap import CLAP as CLAPS

from msclap import CLAP

import yaml


def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


class ExtendedModel(nn.Module):
    def __init__(self, base_model):
        super(ExtendedModel, self).__init__()
        self.base = base_model
        
        # Add new layers herec
        self.new_layer = nn.Linear(in_features=10, out_features=5) 
        
    def forward(self, x):
        x = self.base(x)
        x = self.new_layer(x)
        return x


if __name__ == "__main__":
    # Load your configuration file
    config_path = '../configs/config_2022.yml'  # Replace with your actual file path
    config = load_config(config_path)

    # Load and initialize CLAP
    weights_path = '/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth'
    clap_model = CLAP(weights_path, version = '2022', use_cuda=False)


    clap_instance = CLAPS(
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

    state_dict = torch.load('/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth', map_location=torch.device('cpu'))

    # Assuming you have loaded your checkpoint as follows
    checkpoint = torch.load('/Users/anshumansinha/Desktop/Fall23/CSE8803/Project/CLAP_weights_2022.pth', map_location=torch.device('cpu'))

    # Extract the state dict for the model
    model_state_dict = checkpoint['model']  # 'model' here refers to the key containing the model's state_dict
    clap_instance.load_state_dict(state_dict,strict=False)

    extended_model = ExtendedModel(clap_instance)

    #def forward(self, audio, text): # check

    x = torch.rand(3, 4)

    out = extended_model(x)
    exit()

