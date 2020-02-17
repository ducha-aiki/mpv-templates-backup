import numpy as np
import math
import torch
import torch.nn.functional as F
import typing


def hdist(H: torch.Tensor, pts_matches: torch.Tensor):
    '''Function, calculates one-way reprojection error
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error


    Shape:
        - Input :math:`(3, 3)`, :math:`(B, 4)`
        - Output: :math:`(B, 1)`
    '''
    dist = torch.zeros(pts_matches.size(0),1)
    return dist


def sample(pts_matches: torch.Tensor, num: int=4):
    '''Function, which draws random sample from pts_matches
    
    Return:
        torch.Tensor:

    Args:
        pts_matches: torch.Tensor: 2d tensor
        num (int): number of correspondences to sample

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(num, 4)`
    '''
    sample = torch.zeros(num,4)
    return sample



def getH(min_sample):
    '''Function, which estimates homography from minimal sample
    Return:
        torch.Tensor:

    Args:
        min_sample: torch.Tensor: 2d tensor

    Shape:
        - Input :math:`(B, 4)`
        - Output: :math:`(3, 3)`
    '''
    H_norm = torch.eye(3)
    return  H_norm


def nsamples(n_inl:int , num_tc:int , sample_size:int , conf: float):
    return 0
    


def ransac_h(pts_matches: torch.Tensor, th: float = 4.0, conf: float = 0.99, max_iter:int = 1000):
    '''Function, which robustly estimates homography from noisy correspondences
    
    Return:
        torch.Tensor: per-correspondence Eucledian squared error

    Args:
        pts_matches: torch.Tensor: 2d tensor
        th (float): pixel threshold for correspondence to be counted as inlier
        conf (float): confidence
        max_iter (int): maximum iteration, overrides confidence
        
    Shape:
        - Input  :math:`(B, 4)`
        - Output: :math:`(3, 3)`,   :math:`(B, 1)`
    '''
    Hbest = torch.eye(3)
    inl = torch.zeros(pts_matches.size(0),1) > 0
    return Hbest, inl




