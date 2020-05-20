
from klt import track_klt
from klt_params import params 

import torch 
import kornia 
import cv2
import numpy as np
import os.path

translations = [ [0, 0], 
                 [0.5, 0], 
                 [0, 0.5], 
                 [0.75, -0.2], 
                 [-0.7, 1.5] ]; 

def get_test_data(idx): 
    fname = os.path.join('test_data', 'toy_im_%01i.png' % idx)
    img = cv2.imread(fname)
    img = kornia.color.bgr_to_grayscale(kornia.image_to_tensor(img,False))/255.0
    return img, translations[idx] 


pars = params['default'] 

# read template image: 
img_prev, _ = get_test_data(0) 
# define a point in it:
#   coordinates:
xs = torch.tensor((32.0, 32.0)).reshape(1, -1)
#   id: 
point_ids = torch.tensor((0,))
# change patch size: 
pars.klt_window = 20

# read next image: 
test_idx = 3 # possible values: 0..4 (0 = no shift)
img_next, true_translation = get_test_data(test_idx)
# call the tracker: 
xs_new, point_ids_new = track_klt(img_prev, img_next, xs, point_ids, pars)
estim_translation = xs_new - xs

print('True translation:      %1.2f, %1.2f' % (true_translation[0], true_translation[1]))
print('Estimated translation: %1.2f, %1.2f' % (estim_translation[0, 0], estim_translation[0, 1]))
