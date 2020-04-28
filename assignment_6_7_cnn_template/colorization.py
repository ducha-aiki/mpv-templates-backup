import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List
from PIL import Image
import os
from tqdm import tqdm_notebook as tqdm
from time import time
import torchvision as tv


    
class Unet(nn.Module):
    ''''''
    def __init__(self):
        super(Unet, self).__init__()
        return
    def forward(self, input):
        out = input
        return out

class UnetFromPretrained(nn.Module):
    def __init__(self, encoder):
        super(UnetFromPretrained, self).__init__()
        return
    def forward(self, input):
        out = input
        return out
        
class ContentLoss(nn.Module):
    """"""
    def __init__(self, arch = 'alexnet', layer_id = 11):
        super(ContentLoss, self).__init__()
        return
    def forward(self, input, label):
        loss = input.mean()
        return loss



