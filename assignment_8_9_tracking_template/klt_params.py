from types import SimpleNamespace
from copy import deepcopy 

# This file enables you to load different sets of parameters 
# The set to start with is called default: 
#
# from klt_params import params 
# pars = params['default']
# 
# You can easily add your own sets of parameters for quick switch 
# between parameter sets. 

params = {}

# DEFAULT parameters
pars = SimpleNamespace()

# dataset and filenames-related:
pars.frameN         = 284

# KLT pars:
pars.klt_window     = 10 
pars.klt_max_iter   = 20 
pars.klt_stop_thr   = 0.02**2
pars.klt_sigma_d    = 1.3

pars.harris_thr      = 0.03**4
pars.harris_sigma_d  = 1.4
pars.harris_sigma_i  = 2.5

# we'll call this default, although it is in no way the "best"
params['default'] = pars

# want to change some parameters? Do and name it as you like: 
params['example'] = deepcopy(pars)
params['example'].klt_window = 20 
params['example'].klt_sigma_d = 2.0
