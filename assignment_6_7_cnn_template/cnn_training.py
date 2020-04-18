import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia as K
import typing
from typing import Tuple, List
from PIL import Image
import os
from tqdm import tqdm
from time import time


def get_dataset_statistics(dataset: torch.utils.data.Dataset) -> Tuple[List, List]:
    '''Function, that calculates mean and std of a dataset (pixelwise)'''
    mean = [0., 0., 0.]
    std = [1.0, 1.0, 1.0]
    return mean, std     


class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride = 2, padding=3, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride = 2, padding=2, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride = 1, padding=1, bias = False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride = 1, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.clf = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(512, num_classes))
        return
    
    def forward(self, input):
        x = self.features(input)
        return self.clf(x)


def weight_init(m: nn.Module):
    '''Function, which fills-in weights and biases for convolutional and linear layers'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
     pass #do something
    return

def train_single_epoch(model: torch.nn.Module,
                       train_loader: torch.utils.data.DataLoader,
                       optim: torch.optim.Optimizer,
                       loss_fn: torch.nn.Module) -> torch.nn.Module:
    '''Function, which runs training over a single epoch in the dataloader and returns the model'''
    model.train()
    return model

def lr_find(model, train_dl, loss_fn, min_lr=1e-7, max_lr=100, steps = 50):
    '''Function, which runs the mock training over with different learning rates'''
    lrs = np.ones(steps)
    losses = np.ones(steps)
    return losses, lrs


def validate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    '''Function, which runs the module over validation set and returns accuracy'''
    print ("Starting validation")
    acc = 0
    return acc



class TestFolderDataset(torch.utils.data.Dataset):
    ''''''
    def __init__(self, folder_name, transform = None):
        return
    def __getitem__(self, index):
        img = Image.new('RGB', (128, 128))
        return img
    def __len__(self):
        ln = 0
        return ln
        

def get_predictions(model, test_dl):
    '''Outputs prediction over test data loader'''
    out = torch.zeros(len(test_dl)).long()
    return out
