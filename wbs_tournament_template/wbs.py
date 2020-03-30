import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
import typing


def get_MAE_imgcorners(h,w, H_gt, H_est):
    '''Example of usage:
    H_gt = np.loadtxt(Hgt)
    img1 = K.image_to_tensor(cv2.imread(f1,0),False)/255.
    img2 = K.image_to_tensor(cv2.imread(f2,0),False)/255.
    h = img1.size(2)
    w = img1.size(3)
    H_out = matchImages(img1,img2)
    MAE = get_MAE_imgcorners(h,w,H_gt, H_out.detach().cpu().numpy())
    '''
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H_est).squeeze(1)
    dst_GT = cv2.perspectiveTransform(pts, H_gt).squeeze(1)
    error = np.abs(dst - dst_GT).sum(axis=1).mean()
    return error

# After per-pair MAE calculation, it is reduced as following:
# acc = []
# for th in [0.1, 1.0, 2.0, 5.0, 10., 15.]:
#    A = (np.array(MAEs) <= th).astype(np.float32).mean()
#    acc.append(A)
# MAA = np.array(acc).mean()
# print (MAA)



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
