import os
from oggm import cfg, tasks, graphics, workflow
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import geopandas as gpd

cfg.initialize()
cfg.set_divides_db()
cfg.PARAMS['use_multiprocessing'] = False
# set dem resolution to 40 meters
cfg.PARAMS['grid_dx_method'] = 'fixed'
cfg.PARAMS['fixed_dx'] = 40
cfg.PARAMS['border'] = 10

# The commands below are just importing the necessary modules and functions
# Plot defaults
plt.rcParams['figure.figsize'] = (9, 6)  # Default plot size
# Scientific packages
import numpy as np
# Constants
import copy
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.models.massbalance import LinearMassBalanceModel
from oggm.core.models.flowline import FluxBasedModel
from oggm.core.models.flowline import VerticalWallFlowline, TrapezoidalFlowline, ParabolicFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import pickle
FlowlineModel = partial(FluxBasedModel, inplace=False)

def delta_h_parametrisation(elevation, model):
    grid_len = int(model.length_m/model.fls[-1].dx_meter)

    glacier_elevation = elevation[:grid_len]
    print(glacier_elevation, len(glacier_elevation))
    # normalize elevation

    n_elevation = (np.max(glacier_elevation)-glacier_elevation )/\
                  (np.max(glacier_elevation)-np.min(glacier_elevation))
    print(n_elevation)
    # classification
    if model.area_km2 >= 20:
        a = -0.02
        b = 0.12
        c = 0
        gamma = 6
    elif 5 <= model.area_km2 < 20:
        a = -0.05
        b = 0.19
        c = 0.01
        gamma = 4
    elif model.area_km2 < 5:
        a = -0.3
        b = 0.6
        c = 0.09
        gamma = 2
    delta_h = (n_elevation+a)**gamma+b*(n_elevation+a)+c
    print(np.zeros(len(elevation)-grid_len))
    ice_change = np.insert(delta_h,len(delta_h),np.zeros(len(elevation)-grid_len))
    plt.figure(0)
    plt.plot(model.fls[-1].bed_h,'k--', label='glacier bed')
    plt.plot(elevation, label= '$h_0$')
    plt.plot(model.fls[-1].surface_h+100*ice_change, label='glacier advance: $h_1 = h_0 + 100*\Delta h$')
    plt.plot(model.fls[-1].surface_h-100*ice_change, label='glacier retreat: $h_1 = h_0 - 100*\Delta h$')
    plt.grid(linestyle=':')
    plt.xlabel('Grid points')
    plt.ylabel('Altitude (m)')
    plt.title('Glacier change calculated with $\Delta h$-parameterization (one step)')
    plt.legend(loc='best');

    plt.figure(1)
    plt.gca().invert_yaxis()
    plt.plot(n_elevation, delta_h, color='k', label='$\Delta h=(h_r-0.3)²+0.6(h_r-0.3)+0.09$')

    plt.ylabel('Normalized ice thickness change')
    plt.xlabel('Normalized elevation range $h_r$')
    plt.title('$\Delta h $-parameterization for simple flowline model (small glacier)')
    plt.legend(loc='best')
    plt.show()

#glacier  bed
# This is the bed rock, linearily decreasing from 3000m altitude to 1000m, in 200 steps
nx = 200
bed_h = np.linspace(3400, 1400, nx)
# At the begining, there is no glacier so our glacier surface is at the bed altitude
surface_h = bed_h
# Let's set the model grid spacing to 100m (needed later)
map_dx = 100

# The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
widths = np.zeros(nx) + 3.

# Define our bed
init_flowline = VerticalWallFlowline(surface_h=surface_h, bed_h=bed_h, widths=widths, map_dx=map_dx)

# ELA at 3000m a.s.l., gradient 4 mm m-1
mb_model = LinearMassBalanceModel(3000, grad=4)
annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR


# The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
model = FlowlineModel(init_flowline, mb_model=mb_model, y0=0)
model.run_until(150)
initial = copy.deepcopy(model.fls[-1].surface_h)
model.run_until(300)
surface_h_300 = copy.deepcopy(model.fls[-1].surface_h)

# test huss parametrisation
delta_h_parametrisation(surface_h_300, model)
'''
plt.figure(1)
# Plot the initial conditions first:
plt.plot(init_flowline.bed_h, color='k', label='Bedrock')
plt.plot(initial,color='r', label='Initial glacier')
# The get the modelled flowline (model.fls[-1]) and plot it's new surface
plt.plot(surface_h_300, color='b', label='after 300 years')

plt.xlabel('Grid points')
plt.ylabel('Altitude (m)')
plt.legend(loc='best');
plt.show()
'''

