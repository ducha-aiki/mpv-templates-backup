import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import typing


def matchImages(timg1: torch.Tensor,
                timg2: torch.Tensor):
    r"""Returns the homography, which maps image 1 into image 2
    Args:
        timg1: torch.Tensor: 4d tensor of shape [1x1xHxW]
        timg2: torch.Tensor: 4d tensor of shape [1x1xH2xW2]
    Returns:
        H: torch.Tensor: [3x3] homography matrix
        
    Shape:
      - Input: :math:`(1, 1, H, W)`, :math:`(1, 1, H, W)`
      - Output: :math:`(3, 3)`
    """
    return torch.eye(3)
