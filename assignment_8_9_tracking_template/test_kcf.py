
from kcf import kcf_init, track_kcf
from kcf_params import params

import cv2
import numpy as np
import os.path

translations = [ [0, 0], 
                 [3, 0], 
                 [0, 3], 
                 [7, -3], 
                 [-9, 2] ]; 

def get_test_data(idx): 
    fname = os.path.join('test_data_kcf', 'toy_im_%01i.png' % idx)
    img = cv2.imread(fname).astype(float)/255.0
    img = img[:,:,[2,1,0]]
    return img, translations[idx] 


pars = params['default'] 
# do not use cosine weighting in this test: 
pars.envelope_type = 'uniform'

# read template image: 
img_prev, _ = get_test_data(0) 
# define a bbox in it: 
bbox = [10, 10, 44, 44] # x, y, w, h

# initialize the tracker
S = kcf_init(img_prev, bbox, pars)

# read next image: 
test_idx = 3 # possible values: 0..4 (0 = no shift)
img_next, true_translation = get_test_data(test_idx)

# call the tracker: 
estim_translation = track_kcf(img_next, S, pars)

print('True translation:      %1.2f, %1.2f' % (true_translation[0], true_translation[1]))
print('Estimated translation: %1.2f, %1.2f' % (estim_translation[0], estim_translation[1]))
