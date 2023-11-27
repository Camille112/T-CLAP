import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import pandas as pd
from pathlib import Path
import glob, re, os
import cardiac_ml_tools as ct
import os
import pandas as pd

[{
    "text": "pig before helicopter",
    "path-a": "2-158746-B-2.wav",
    "path-b": "4-193480-A-40.wav",
    "path": "TemporalAudio/158746-before-193480.wav"
},
{
    "text": "helicopter before pig",
    "path-a": "2-158746-B-2.wav",
    "path-b": "4-193480-A-40.wav",
    "path": "TemporalAudio/193480-before-158746.wav"
},
{
    "text": "pig while helicopter",
    "path-a": "2-158746-B-2.wav",
    "path-b": "4-193480-A-40.wav",
    "path": "TemporalAudio/158746-while-193480.wav"
}]


def read_data(data_dir: Union[str, Path] = "data"):

    csv_file = 'data.csv' # path
    df = pd.read_csv(csv_file)
    selected_df = df[['filename1', 'category1']]
    print(selected_df)
    
    file_pairs = ct.read_data_dirs(data_dirs)
    
    lf1 = np.load(file_pairs[0][0]) # (500, 10)
    lf1 = ct.get_standard_leads(lf1) # (500, 12)

    lf2 = np.load(file_pairs[0][1]) # (75, 500)
    lf2 = ct.get_activation_time(lf2) # (75, 1)

    data_book= []
    count = 0

    for lf1,lf2 in file_pairs:
        
        x1 = (np.load(lf1))
        x1 = ct.get_standard_leads(x1)
        
        x2 = (np.load(lf2))
        x2 = ct.get_activation_time(x2)
        
        data_book.append((x1, x2))
        count+=1

        if(count>10): # remove these lines while implementing.
            break
    
    df = pd.DataFrame(data_book, columns=['leads', 'act'])
    return df

class Custom_dataset(Dataset):
    
    def __init__(self, annotations_file):
        self.img_labels = annotations_file

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        x = self.img_labels.iloc[idx, 0]
        x = torch.tensor(x, dtype=torch.float32) # Pytorch default for nn.models
        y = self.img_labels.iloc[idx, 1]
        y = torch.tensor(y, dtype =torch.float32) # Pytorch default for nn.models
        return x, y

if __name__ == "__main__":

    data = read_data()
    ds_data = Custom_dataset(data)
    generator = torch.Generator().manual_seed(42)
    train_ds,val_ds,test_ds = torch.utils.data.random_split(ds_data, [0.8, 0.15, 0.05], generator=generator)
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    # Report split sizes
    print('Training set has {} instances'.format(len(train_ds)))
    print('Validation set has {} instances'.format(len(val_ds)))
    print('Test set has {} instances'.format(len(test_ds)))

    for i,j in training_loader:
        print('i : ',i)
        print('i : ',i.shape)
        print('j : ',j)
        print('j : ',j.shape)
        break

