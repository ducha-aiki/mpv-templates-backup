import pdb, torch
import time
import os
import math
from torchvision import transforms
import cv2
import kornia
import numpy as np
from IPython.display import HTML, clear_output
import IPython.display
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.image as mpimg
from celluloid import Camera 
import numpy
import ipywidgets as widgets
import tkinter as tk
from tkinter import ttk
from ipywidgets import interact, interactive, fixed, interact_manual
import mpl_interactions.ipyplot as iplt
import time
from kornia.feature import get_laf_orientation, set_laf_orientation, extract_patches_from_pyramid

FULL_ROTATION = 360
SIZE_IMG = 150

def play_with_angle(img, A, orientation_estimation):
  """
  Interactive visualization(with the slider) of working of your orientation_estimation function.

  Args:
    patch: (torch.Tensor) 
    orientation_estimation: estimator function

  Returns:
    nothing, but as side affect patches are shown  
  """
  laf = A[:,:2,:].reshape(1,1,2,3)
  orig_angle = get_laf_orientation(laf)
  patch = extract_patches_from_pyramid(img, laf)[0]
  
  patch = kornia.tensor_to_image(patch)
  fig, ax = plt.subplots(1, 3, figsize=(12, 4))
  ax1 = ax[0].imshow(patch, cmap='gray')
  ax2 = ax[1].imshow(patch, cmap='gray')
  ax3 = ax[2].imshow(patch, cmap='gray')

  ax[0].set_title("Rotated patch")
  ax[1].set_title("user normalized patch")
  ax[2].set_title("kornia normalized patch")
   
    
  plt.close()

  slider = widgets.FloatSlider(value=0, min=0, max=360, step=1, description="Angle:")
  widgets.interact(img_viz, img=fixed(img), A=fixed(A), orientation_estimation=fixed(orientation_estimation), fig=fixed(fig), ax1=fixed(ax1), ax2=fixed(ax2), ax3=fixed(ax3), alfa=slider)


# helper function. It is called as a parametr of widgets.interact()
def img_viz(img: torch.tensor, A: torch.tensor, orientation_estimation, fig, ax1, ax2, ax3, alfa=0):
  alfa = alfa
  laf = A[:,:2,:].reshape(1,1,2,3)
  orig_angle = get_laf_orientation(laf)
  patch = extract_patches_from_pyramid(img, laf)
  angle = torch.tensor([np.float32(alfa)])
  laf_current = set_laf_orientation(laf, alfa + orig_angle)
    
  patch_rotated = extract_patches_from_pyramid(img, laf_current)[0]

  grad_ori = kornia.feature.orientation.PatchDominantGradientOrientation(32)
  estimated_angle = -orientation_estimation(patch_rotated).reshape(-1)
  estimated_angle_kornia = grad_ori(patch_rotated).reshape(-1)

  prev_angle = get_laf_orientation(laf_current)
  laf_out_user = set_laf_orientation(laf_current, torch.rad2deg(estimated_angle) + prev_angle)
  laf_out_kornia = set_laf_orientation(laf_current, torch.rad2deg(estimated_angle_kornia) + prev_angle)

  patch_out = extract_patches_from_pyramid(img, laf_out_user).reshape(1,1,32,32)
  patch_out_kornia = extract_patches_from_pyramid(img, laf_out_kornia).reshape(1,1,32,32)

  img1 = kornia.tensor_to_image(patch_rotated)
  img2 = kornia.tensor_to_image(patch_out)
  img3 = kornia.tensor_to_image(patch_out_kornia)
    

  ax1.set_data(img1)
  ax2.set_data(img2)
  ax3.set_data(img3)
  display(fig)
  plt.close()
