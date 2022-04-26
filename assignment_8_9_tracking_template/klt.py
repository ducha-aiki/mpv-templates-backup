from imagefiltering import spatial_gradient_first_order, extract_affine_patches
from ransac import ransac_h

import os.path
import kornia 
import cv2
import numpy as np
import torch
import typing
from types import SimpleNamespace
import matplotlib.pyplot as plt

def read_image(idx): 
    fname = os.path.join('data', '%03i.jpg' % idx)
    img = cv2.imread(fname)
    img = kornia.color.bgr_to_grayscale(kornia.image_to_tensor(img.astype(np.float32),False))/255.0
    return img

def get_patches_subpix(img: torch.Tensor, 
                       xs: torch.Tensor, 
                       window_hsize: int): 
    """
    Extract square patches from image img. 
    The patches are defined by square centers xs, and extent given by 
    window_hsize. 

    
    Args:
       img: torch.Tensor, size (1, 1, H, W): image
       xs:  torch.Tensor, size (N, 2): coordinates of centers of patches to extract, 
       window_hsize: (int) half of the square size.
    
       In terms of previously implemented function extract_affine_patches, 
       ext = window_hsize, and 
       PS = window_hsize*2+1 .

    Returns:
       patches: torch.Tensor, size (N, 1, PS, PS)
    """

    # YOUR_IMPLEMENTATION_START
    PS = window_hsize*2+1
    return torch.rand(len(xs), 1, PS, PS)
    # YOUR_IMPLEMENTATION_END



def klt_compute_updates(Ts: torch.Tensor, 
                        img_next: torch.Tensor, 
                        img_next_gradient: torch.Tensor, 
                        xs: torch.Tensor, 
                        ps: torch.Tensor, 
                        window_hsize: int, 
                        ):
    """
    Compute KLT update. 
    
    Args: 
       Ts: torch.Tensor, size (N, 1, L, L): N patches extracted from template image   
           (note: L == 2*windows_hsize+1)
       img_next: torch.Tensor, size (1, 1, H, W): next (target) image 
       img_next_gradient: torch.Tensor, size (1, 1, 2, H, W): gradient, as computed 
                on img_next using function spatial_gradient_first_order 
       xs: torch.Tensor, size (N, 2): centers of patches Ts in the template image. 
                Each row stores the x, y coordinates of the center of the respective patch. 
       ps: torch.Tensor, size (N, 2): current estimates of shift between template and target images,
                for all tracked patches. The centers of tracked patches in the target image 
                are computed as xs+ps. 
       window_hsize: (int): half of the square side of the patches.

    Returns: 
       dps: torch.Tensor, size (N, 2): update of the estimated position of tracked patches 
            in the target image; the updated shifts are computed as 
            ps <- ps + dps
    """

    # YOUR_IMPLEMENTATION_START
    N = len(xs) # number of patches 
    dps = torch.zeros(N, 2) # allocation of result
    # YOUR_IMPLEMENTATION_END

            
    return dps

def track_klt(img_prev: torch.Tensor, 
              img_next: torch.Tensor, 
              xs: torch.Tensor, 
              point_ids: torch.Tensor, 
              pars: SimpleNamespace): 

    window_hsize = pars.klt_window
    Ts = get_patches_subpix(img_prev, xs, window_hsize)
    img_next_gradient = spatial_gradient_first_order(img_next, pars.klt_sigma_d)
    N = len(xs)

    ps = torch.zeros(N, 2) # at start, estimate of shift is 'no shift'

    iter_count = 0 
    
    while iter_count < pars.klt_max_iter:
        dps = klt_compute_updates(Ts, img_next, img_next_gradient, xs, ps, window_hsize)
        ps += dps 
        iter_count += 1 
  
    # if an element of dps is smaller then threshold then the point has converged.
    dps_norm2 = (dps**2).sum(axis=1)
    idx = (dps_norm2 <= pars.klt_stop_thr).nonzero().squeeze() 
    # for further tracking, keep just the converged points: 
    xs_new = xs[idx] + ps[idx] 
    point_ids_new = point_ids[idx] 
    return xs_new, point_ids_new 

def compute_and_show_homography(pts_fname = 'tracked_points.pt', result_fname = 'result.pdf'): 

    def show_result(im, frame_idx, xs, xs_bbox, inl):
        plt.imshow(im[0, 0, :, :], cmap='gray')
        plt.title('frame: %i' % (frame_idx,))
        plt.plot(xs_bbox[0, [0,1,2,3,0]], xs_bbox[1, [0,1,2,3,0]], 'r')
        plt.plot(xs[inl.logical_not(),0], xs[inl.logical_not(),1], 'kx')
        plt.plot(xs[inl,0], xs[inl,1], 'rx')


    points_frame_first, points_frame_last = torch.load(pts_fname)
    
    # indices of first and last frames (but note that frame_idx0 
    # is assumed to be 0 and positions of corners of the leaflet are 
    # hard-coded below): 
    frame_idx0, frame_idx1 = points_frame_first[0], points_frame_last[0]
    # point coordinates in the first and last frames: 
    xs0, xs1 = points_frame_first[1], points_frame_last[1] 
    # identities (indices) of points in the first and last frames:
    ids0, ids1 = points_frame_first[2], points_frame_last[2]
 
    # find all points with the same indices and put their 
    # coordinates to variable correspondences in the format 
    # expected by function ransac_h
    # that is, each row is [x0, y0, x1, y1]
    # where (x0, y0) is the coordinate of a point in the 1st frame
    # and   (x1, y1) is the coordinate of the corresponding point
    # in the last frame.

    # YOUR_IMPLEMENTATION_START
    correspondences = torch.cat( (xs1, xs1), 1)
    # YOUR_IMPLEMENTATION_END


    Hbest, inl = ransac_h(correspondences, 3.0, 0.99)

    # 4 corners of the leaflet in the first frame: 
    xy0 = torch.tensor([[58.3, 48.4],
                       [316.0, 40.7],
                       [328.1, 413.6],
                       [15.3, 423.1]])

    # transform it by the estimated homography: 
    xy0 = torch.cat((xy0, torch.ones(4,1)), 1).t()
    xy1 = torch.mm(Hbest, xy0)
    xy1[:2, :] /= xy1[2,:]

    # read the first and last frames of the sequence: 
    img0  = read_image(frame_idx0)
    img1  = read_image(frame_idx1)

    inl = inl.squeeze()

    plt.figure(3)
    plt.suptitle('#correspondences: %i, #inliers(in red): %i' % (len(correspondences), inl.sum()))
    plt.subplot(1, 2, 1)
    show_result(img0, 0, correspondences[:,:2], xy0, inl)
    plt.subplot(1, 2, 2)
    show_result(img1, frame_idx1, correspondences[:,2:], xy1, inl)

    # save this figure:
    plt.savefig(result_fname)
