# Built ins
import os
import copy
from functools import partial

# External libs
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import pickle

# locals
import salem
from oggm import cfg, workflow, tasks, utils
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import PastMassBalance, RandomMassBalance, LinearMassBalance
from oggm.core.flowline import FluxBasedModel,RectangularBedFlowline
FlowlineModel = partial(FluxBasedModel, inplace=False)
from scipy.interpolate import UnivariateSpline

def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    old_indices = np.arange(0, len(array))
    new_length = mx
    new_indices = np.linspace(0, len(array) - 1, new_length)
    spl = UnivariateSpline(old_indices, array, k=1, s=0)
    new_array = spl(new_indices)
    return new_array

# glacier  bed
# This is the bed rock, linearily decreasing from 3000m altitude to 1000m, in 200 steps
nx = 200
bed_h = np.linspace(3400, 1400, nx)
# At the begining, there is no glacier so our glacier surface is at the bed altitude

# Let's set the model grid spacing to 100m (needed later)
map_dx = 100

# The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
widths = np.zeros(nx) + 3.
# Define our bed
init_flowline = RectangularBedFlowline(surface_h=bed_h, bed_h=bed_h,
                                     widths=widths, map_dx=map_dx)
# ELA at 3000m a.s.l., gradient 4 mm m-1
mb_model = LinearMassBalance(3000, grad=4)
model = FlowlineModel(init_flowline, mb_model=mb_model, y0=0)
model.run_until(300)
original = copy.deepcopy(model)

back = FlowlineModel(original.fls, mb_model=mb_model, y0=300)
back.run_until_back(299.94)
back.run_until(304)
#back.run_until_back(299.94)

plot_dir = '/home/juliaeis/Dropbox/geteilt/OGGM_workshop_2018/plots'
plt.figure()
#plt.plot(original.fls[-1].surface_h, label = 't=300')
plt.plot(back.fls[-1].surface_h, label = 't=299.9')
plt.plot(original.fls[-1].bed_h,'k', label='Bedrock')
plt.xlabel('Distance along the Flowline (m)',size=12)
plt.ylabel('Altitude (m)',size=12)
plt.xlim(0,120)
plt.ylim(2300,3600)
plt.legend(loc='best',fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
#plt.savefig(os.path.join(plot_dir, 'backward_3.png'),dpi=200)
plt.show()