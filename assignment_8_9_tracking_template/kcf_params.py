from types import SimpleNamespace
from copy import deepcopy 

# This file enables you to load different sets of parameters 
# The set to start with is called default: 
#
# from kcf_params import params 
# pars = params['default']
# 
# You can easily add your own sets of parameters for quick switch 
# between parameter sets. 

params = {}

# DEFAULT parameters
pars = SimpleNamespace()

# dataset and filenames-related:
pars.frameN         = 284

# KCF pars:
pars.rbf_sigma = 2    # RBF Gauss kernel bandwidth
pars.gamma     = 0.075  # adaptation rate 
pars.lam       = 1e-4   # regularization constant
pars.kernel_type = 'rbf'
#pars.kernel_type = 'linear'
pars.envelope_type = 'cos' # cosine window to mitigate boundary effects in FFT
#pars.envelope_type = 'uniform'

# we'll call this default, although it is in no way the "best"
params['default'] = pars

# want to change some parameters? Do and name it as you like: 
params['example'] = deepcopy(pars)
params['example'].rbf_sigma = 3.0
