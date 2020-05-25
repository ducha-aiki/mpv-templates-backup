
from kcf import read_image, kcf_init, track_kcf
from kcf_params import params 
import numpy as np 
import matplotlib.pyplot as plt
import torch

# filename for saving the patch shifts over the entire sequence: 
pts_fname = 'kcf_translations.pt'
# filename for saving the figure showing the coordinates of path of the tracked patch:
result_fname = 'result_kcf.pdf'

DISPLAY=True

pars = params['default'] 

# read the 1st image 
img = read_image(0) 
bbox = [110, 160, 80, 65] # x, y, w, h

S = kcf_init(img, bbox, pars)



shifts = [] # for storing the shifts
for k in range(1, pars.frameN): 
    # read next image
    img_next = read_image(k) 
    # call the tracker
    dx, dy = track_kcf(img_next, S, pars)
    shifts.append((dx, dy)) 
    
    if DISPLAY:
        plt.figure(1) 
        plt.clf()
        plt.suptitle('frame %i, dx=%i, dy=%i' % (k, dx, dy) )

        plt.subplot(2,2,1)
        plt.imshow(img_next)
        plt.plot([S.x, S.x+S.w, S.x+S.w, S.x, S.x], [S.y, S.y, S.y+S.h, S.y+S.h, S.y], 'b')

        plt.subplot(2,2,2) 
        plt.imshow(S.responses) 
        # also show the reference center of the response: 
        plt.plot(S.cx, S.cy, 'rx') 
        plt.title('current response')


        plt.subplot(2,2,3)     
        plt.imshow(S.patch_next * S.envelope) 
        plt.title('current patch, w. by envelope')

        plt.subplot(2,2,4)     
        plt.imshow(S.x_train) 
        plt.title('adapted training patch') 


        plt.draw()
        plt.pause(0.01)

# save points in the 1st and last frames: 
torch.save([shifts,], pts_fname)

# show the path of the tracked patch: 
shifts = torch.load(pts_fname)[0]
xy = np.array(shifts).cumsum(axis=0)

if DISPLAY: 
    plt.figure(2)
    plt.clf() 
    plt.plot(xy[:,0], xy[:,1], 'r')
    plt.title('path of the tracked patch')
    plt.savefig(result_fname, transparent = True)
