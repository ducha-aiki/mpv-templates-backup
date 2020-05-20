from local_detector import harris
from klt import track_klt, read_image, compute_and_show_homography
from klt_params import params 

# filename for saving the tracked points: 
pts_fname = 'tracked_points.pt'
# filename for saving the figure demonstrating estimated homography:
result_fname = 'result.pdf'

pars = params['default']
# for testing, you may want to set frameN to smaller number
# pars.frameN = 50

import kornia 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

DISPLAY=True

def show_frame(im, frame_idx, xs):
    plt.imshow(im[0, 0, :, :], cmap='gray')
    plt.plot(xs[:, 0], xs[:,1], 'rx')
    plt.title('frame: %i, #points: %i' % (frame_idx, len(xs)))
    plt.draw()
    plt.pause(0.001)


# read template image:
img_template  = read_image(0)
# detect Harris points in it: 
harris_points = harris(img_template, pars.harris_sigma_d, pars.harris_sigma_i, pars.harris_thr)
# use them as centers of patches for tracking: 
xs = harris_points[:, [3, 2]].float() 

if DISPLAY:
    plt.close(1) 
    plt.close(2) 
    plt.close(3) 
    plt.figure(1)
    show_frame(img_template, 0, xs) 

# add point id's:
pointsN = len(xs)
point_ids = torch.arange(pointsN)
# store them for later: 
points_frame_first = 0, xs, point_ids  # 0=frame_number

img_prev = img_template
for k in range(1, pars.frameN): 
    img_next = read_image(k)
    xs_new, point_ids_new = track_klt(img_prev, img_next, xs, point_ids, pars)

    # prepare for next iteration: 
    img_prev = img_next
    xs, point_ids = xs_new, point_ids_new 

    if DISPLAY:
        plt.figure(2)
        plt.cla() 
        show_frame(img_next, k, xs) 


# store points at the end of sequence: 
points_frame_last = pars.frameN-1, xs, point_ids

# save points in the 1st and last frames: 
torch.save([ points_frame_first, points_frame_last ], pts_fname)

# load points, estimate homography and show the result: 
compute_and_show_homography(pts_fname = pts_fname, result_fname = result_fname)
