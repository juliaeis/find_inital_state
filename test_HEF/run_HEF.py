from scipy.optimize import minimize
# Scientific packages
import numpy as np
# Constants
from oggm import graphics,cfg
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.models.massbalance import LinearMassBalanceModel
from oggm.core.models.flowline import FluxBasedModel
from oggm.core.models.flowline import VerticalWallFlowline, \
    TrapezoidalFlowline, ParabolicFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import pickle
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
FlowlineModel = partial(FluxBasedModel, inplace=False)
from oggm.core.models.massbalance import ConstantMassBalanceModel
import math
import multiprocessing as mp
import copy
from oggm.utils import get_demo_file
import geopandas as gpd
global gdir_hef

plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    old_indices = np.arange(0, len(array))
    new_length = mx
    new_indices = np.linspace(0, len(array) - 1, new_length)
    spl = UnivariateSpline(old_indices, array, k=1, s=0)
    new_array = spl(new_indices)
    return new_array


def run_model(fls,ice_thick):
    nx = fls[-1].nx
    today_model = ConstantMassBalanceModel(gdir_hef, y0=1985)
    fls[-1].surface_h = rescale(ice_thick, nx)+fls[-1].bed_h
    commit_model = FluxBasedModel(fls, mb_model=today_model, glen_a=cfg.A)
    return commit_model


def oggm_length_m(fls):
    thick = fls[-1].surface_h-fls[-1].bed_h
    # We define the length a bit differently: but more robust
    pok = np.where(thick > 0.)[0]
    return len(pok)* fls[-1].dx * fls[-1].map_dx


def min_length_m(fls):
    thick = fls[-1].surface_h-fls[-1].bed_h
    # We define the length a bit differently: but more robust
    pok = np.where(thick < 3.)[0]
    return pok[0]* fls[-1].dx * fls[-1].map_dx


if __name__ == '__main__':

    cfg.initialize()
    gdir_hef=pickle.load(open('gdir_hef.pkl','rb'))
    fls = pickle.load(open('HEF_fls.pkl', 'rb'))
    ice_thick = fls[-1].surface_h-fls[-1].bed_h

    model=run_model(fls,ice_thick[np.linspace(0,fls[-1].nx-1,15).astype(int)])
    model.run_until(100)
    print(model.fls[-1].bed_h)
    x = np.arange(fls[-1].nx) * fls[-1].dx * fls[-1].map_dx
    plt.figure(0)
    plt.plot(x,fls[-1].bed_h,'k')
    ice_thick=model.fls[-1].surface_h-model.fls[-1].bed_h

    #plt.plot(x,ice_thick+model.fls[-1].bed_h,color='teal')
    #plt.plot(x[np.linspace(0,fls[-1].nx-1,15).astype(int)],ice_thick[np.linspace(0,fls[-1].nx-1,15).astype(int)]+fls[-1].bed_h[np.linspace(0,fls[-1].nx-1,15).astype(int)],color='r')
    plt.plot(x,rescale(ice_thick[np.linspace(0,fls[-1].nx-1,15).astype(int)],fls[-1].nx) +fls[-1].bed_h,color='teal',label='surface_h[15 grid points]')
    plt.plot(x[np.linspace(0,fls[-1].nx-1,15).astype(int)],ice_thick[np.linspace(0,fls[-1].nx-1,15).astype(int)]+fls[-1].bed_h[np.linspace(0,fls[-1].nx-1,15).astype(int)],'o',color='teal')

    plt.plot(x,
             rescale(ice_thick[np.linspace(0, fls[-1].nx - 1, 30).astype(int)],
                     fls[-1].nx) + fls[-1].bed_h, color='orange',
             label='surface_h[15 grid points]')
    plt.plot(x[np.linspace(0, fls[-1].nx - 1, 30).astype(int)],
             ice_thick[np.linspace(0, fls[-1].nx - 1, 30).astype(int)] +
             fls[-1].bed_h[np.linspace(0, fls[-1].nx - 1, 30).astype(int)],
             'o', color='orange')
    plt.plot(x, model.fls[-1].surface_h)
    plt.legend(loc='best')
    plt.xlabel('Distance along the flowline (m)')
    plt.ylabel('Altitude (m)')

    plt.figure(1)
    plt.plot(x,ice_thick,label='ice thickness')
    plt.plot(x[np.linspace(0,fls[-1].nx-1,15).astype(int)],ice_thick[np.linspace(0,fls[-1].nx-1,15).astype(int)],color='teal',label='ice thickness[15 grid points]')
    plt.plot(x[np.linspace(0, fls[-1].nx - 1, 15).astype(int)],ice_thick[np.linspace(0, fls[-1].nx - 1, 15).astype(int)],'o', color='teal')

    plt.plot(x[np.linspace(0, fls[-1].nx - 1, 30).astype(int)],
             ice_thick[np.linspace(0, fls[-1].nx - 1, 30).astype(int)],
             color='orange', label='ice thickness[30 grid points]')
    plt.plot(x[np.linspace(0, fls[-1].nx - 1, 30).astype(int)],
             ice_thick[np.linspace(0, fls[-1].nx - 1, 30).astype(int)], 'o',
             color='orange')
    plt.legend(loc='best')
    plt.xlabel('Distance along the flowline (m)')
    plt.ylabel('Altitude (m)')
    plt.show()