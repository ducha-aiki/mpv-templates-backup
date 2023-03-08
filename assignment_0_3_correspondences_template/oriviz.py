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
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
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


""" The SIFT visualization """



# Visualization 

def reshape_to_bins(siftdesc, spatial_dim, orient_dim):
  b, dim = siftdesc.shape
  return siftdesc.reshape(b, orient_dim, -1).reshape(b, orient_dim, spatial_dim, spatial_dim)

def visualize_sift(desc,  orient_dim=8, spatial_dim=4, title = '', is_absolute_mode =1):
  res = 5*reshape_to_bins(desc, spatial_dim, orient_dim)
  max = -1.0
  angle = 360.0 / orient_dim
  alfa = 0

  # finding the maximum number in the res tensor
  for i in range(0, spatial_dim):
    for j in range(0, spatial_dim):
      for k in range(0, orient_dim):
        if res[:, :, i, j][0][k] < 0:
          sys.exit("Provided tensor cannot contain a negative number!")
        max = res[:, :, i, j][0][k] if (res[:, :, i, j][0][k] > max) else max

  max = float(max)

  # creating the table
  f, ax = plt.subplots(spatial_dim, spatial_dim, figsize=(10, 10), gridspec_kw = {'wspace':0, 'hspace':0})

  if is_absolute_mode == 1:
    f.suptitle(f"{title} bins abs histogram", fontsize=16)
  else:
    f.suptitle(f"{title} bins rel values", fontsize=16)

  plt.subplots_adjust(wspace=0, hspace=0)
  plt.rcParams['axes.linewidth'] = 5

  # colorizing the table 
  for i in range(spatial_dim):
    for j in range(spatial_dim):
      axx = ax
      if spatial_dim != 1:
        axx = ax[i][j]
      axx.spines['bottom'].set_color('green')
      axx.spines['top'].set_color('green')
      axx.spines['right'].set_color('green')
      axx.spines['left'].set_color('green')
      axx.set_xticklabels([])
      axx.set_yticklabels([])


  # drawing the arrows corresponding the elements of the res tensor
  for i in range(0, spatial_dim):
    for j in range(0, spatial_dim):
      for k in range(0, orient_dim):
        if float(res[:, :, i, j][0][k]) == 0:
          alfa += angle
          continue
        ratio = max / float(res[:, :, i, j][0][k])
        if ratio <= 10 or is_absolute_mode == 1:
          ratio = float(res[:, :, i, j][0][k]) if is_absolute_mode == 1 else float(res[:, :, i, j][0][k]) / max
          axx = ax
          if spatial_dim != 1:
            axx = ax[i][j]
          if alfa >= 0 and alfa <= 90:
            axx.add_patch(mpatches.FancyArrowPatch((1, 1), (1 + ratio * math.cos(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item()), 1 + ratio * math.sin(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item())), mutation_scale=5))
          elif alfa > 90 and alfa <= 180:
            tmp = alfa
            alfa = 180 - alfa
            axx.add_patch(mpatches.FancyArrowPatch((1, 1), (1 - ratio * math.cos(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item()), 1 + ratio * math.sin(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item())), mutation_scale=5))
            alfa = tmp         
          elif alfa > 180 and alfa <= 270:
            tmp = alfa
            alfa -= 180
            axx.add_patch(mpatches.FancyArrowPatch((1, 1), (1 - ratio * math.cos(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item()), 1 - ratio * math.sin(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item())), mutation_scale=5))
            alfa = tmp          
          else: 
            tmp = alfa
            alfa = 360 - alfa
            axx.add_patch(mpatches.FancyArrowPatch((1, 1), (1 + ratio * math.cos(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item()),  1 - ratio * math.sin(kornia.geometry.conversions.deg2rad(torch.tensor(alfa)).item())), mutation_scale=5))
            alfa = tmp
        alfa += angle     
      axx.set(xlim=(0, 2), ylim=(0, 2))
      alfa = 0
  return f 
