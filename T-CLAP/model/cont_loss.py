"""Video-Language Contrastive Losses"""

import numpy as np
import torch
from torch import nn


# Loss composer: albegraic combination of losses
# Just pass in the list of losses.

class LossAddition:
    def __init__(self, losses: list) -> None:
        # self.losses = [loss() for loss in losses]
        self.losses = losses
    def __call__(self, **kwds):
        loss = 0.
        for l in self.losses:
            try:
                loss += l(**kwds)
            except:
                print('okay')
        return loss


class T2AContraLoss:

    """NCE for MM joint space, on softmax text2video matrix."""

    def __init__(self, weight=None, alpha=1.0):
        # TODO (huxu): define temperature.
        print('weight',weight.shape) # weight torch.Size([4])
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.alpha = np.log(alpha + 1e-8)
        if isinstance(self.alpha, np.ndarray):
            self.alpha = torch.from_numpy(self.alpha)

    def __call__(self, batch_audio, batch_text, **kargs):
        batch_size = batch_audio.size(0)
        # change device of the weight

        self.loss.weight = self.loss.weight.to(batch_audio.device)
        self.alpha = self.alpha.to(batch_audio.device)
        logits = torch.mm(batch_text, batch_audio.transpose(1, 0))

        '''
        try:
            logits[batch_size // 2:, :batch_size // 2] += self.alpha
            logits[:batch_size // 2, batch_size // 2:] += self.alpha
        except:
            print('its fine')
        '''

        sub_size = batch_size // 3

        try:
            logits[:sub_size, sub_size:2*sub_size] += self.alpha
            logits[:sub_size, 2*sub_size:] += self.alpha

            logits[sub_size:2*sub_size, :sub_size] += self.alpha
            logits[sub_size:2*sub_size, 2*sub_size:] += self.alpha

            logits[2*sub_size:, :sub_size] += self.alpha
            logits[2*sub_size:, sub_size:2*sub_size] += self.alpha
        except:
            print('its fine')
            
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=batch_audio.device)

        return self.loss(logits, targets)


class A2TContraLoss:

    """NCE for MM joint space, with softmax on video2text matrix."""

    def __init__(self, weight=None, alpha=1.0):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.alpha = np.log(alpha + 1e-8)
        if isinstance(self.alpha, np.ndarray):
            self.alpha = torch.from_numpy(self.alpha)

    # pooled_video = batch x embedding , pooled_text = batch x embedding 
    def __call__(self, batch_audio, batch_text, **kargs): 
        
        batch_size = batch_audio.size(0)
        
        # change device of the weight
        self.loss.weight = self.loss.weight.to(batch_audio.device)
        self.alpha = self.alpha.to(batch_audio.device)

        # calculate the logits
        logits = torch.mm(batch_audio, batch_text.transpose(1, 0))

        sub_size = batch_size // 3

        # update the logits
        logits[:sub_size, sub_size:2*sub_size] += self.alpha
        logits[:sub_size, 2*sub_size:] += self.alpha

        logits[sub_size:2*sub_size, :sub_size] += self.alpha
        logits[sub_size:2*sub_size, 2*sub_size:] += self.alpha

        logits[2*sub_size:, :sub_size] += self.alpha
        logits[2*sub_size:, sub_size:2*sub_size] += self.alpha

        # Usual targets from CLIP
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=batch_audio.device)

        return self.loss(logits, targets)