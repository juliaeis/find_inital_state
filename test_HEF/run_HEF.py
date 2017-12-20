from scipy.optimize import minimize
# Scientific packages
import numpy as np
# Constants
from oggm import graphics,cfg
import pandas as pd
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.massbalance import LinearMassBalance
from oggm.core.flowline import FluxBasedModel
from oggm.core.flowline import RectangularBedFlowline, TrapezoidalBedFlowline, ParabolicBedFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import pickle
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
FlowlineModel = partial(FluxBasedModel, inplace=False)
from oggm.core.massbalance import ConstantMassBalance,RandomMassBalance, PastMassBalance
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


def run_model(surface_h):

    nx = y1.fls[-1].nx
    random_climate = pickle.load(open('random_climate_hef.pkl', 'rb'))
    hef_fls = pickle.load(open('hef_y1.pkl', 'rb'))
    surface_h = rescale(surface_h,nx)
    thick = surface_h-hef_fls[-1].bed_h
    # We define the length a bit differently: but more robust
    try:
        pok = np.where(thick < 10)[0]
        surface_h[int(pok[0]):] = hef_fls[-1].bed_h[int(pok[0]):]

    except:
        pass
    hef_fls[-1].surface_h = surface_h
    commit_model = FluxBasedModel(hef_fls, mb_model=random_climate, glen_a=cfg.A,y0=1850)
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

def objfunc(surface_h):

    model = run_model(surface_h)
    s_temp = copy.deepcopy(model.fls[-1].surface_h)
    try:
        model.run_until(1900)
        f = np.sum(abs(model.fls[-1].surface_h - y1.fls[-1].surface_h))**2 #+ \
            #abs(min_length_m(model.fls) - y1.length_m)**2


    except:
        f = np.nan

        #abs(model.area_km2 - y1.area_km2) + \
        #abs(model.volume_km3 - y1.volume_km3)

    #ax1.plot(s_temp)
    #ax2.plot(model.fls[-1].surface_h)
    print(f)
    return f

def con1(surface_h):
    ''' ice thickness greater than zero'''
    model =  run_model(surface_h)
    return model.fls[-1].thick[np.linspace(0,hef_fls[-1].nx-1,len(surface_h)).astype(int)]

def con2(surface_h):
    ''' ice thickness smaller than 1000'''
    model = run_model(surface_h)
    return -model.fls[-1].thick[np.linspace(0,hef_fls[-1].nx-1,len(surface_h)).astype(int)]+1000

def con3(surface_h):
    '''last pixel has to be zero ice thickness'''
    model = run_model(surface_h)
    return -model.fls[-1].thick[-1]

def con4(surface_h):
    '''glacier change not as much at the head'''
    return 10-(abs(y1.fls[-1].surface_h[0]-surface_h[0]))

def con5(surface_h):
    model = run_model(surface_h)
    surface_h = model.fls[-1].surface_h[np.linspace(0,hef_fls[-1].nx-1,len(surface_h)).astype(int)]
    bed_h = model.fls[-1].bed_h[np.linspace(0, hef_fls[-1].nx - 1, len(surface_h)).astype(int)]
    con_array = np.zeros(len(surface_h))
    for index in range(1,len(surface_h)):
        if (surface_h[index]!= bed_h[index])and(surface_h[index]> surface_h[index-1]):
            con_array[index]=surface_h[index-1]- surface_h[index]
    return con_array

def parallel(rhoberg,x0,cons):
    res = minimize(objfunc, x0, method='COBYLA', tol=1e-04, constraints=cons,
                   options={'maxiter': 5000, 'rhobeg': rhoberg})
    if res.success:
        return res.x


if __name__ == '__main__':
    global all_shapes

    f, ax = plt.subplots(2, sharex=True)

    all_shapes = []
    cfg.initialize()
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
    # get gdir
    gdir_hef=pickle.load(open('gdir_hef.pkl','rb'))


    # get climate model
    random_climate = PastMassBalance(gdir_hef)
    pickle.dump(random_climate,open('random_climate_hef.pkl','wb'))
    #random_climate = pickle.load(open('random_climate_hef.pkl','rb'))

    hef_fls = pickle.load(open('hef_y1.pkl', 'rb'))

    commit_model = FluxBasedModel(hef_fls, mb_model=random_climate, glen_a=cfg.A,y0=1850)

    fls_y0 = copy.deepcopy(commit_model.fls)

    commit_model.run_until(1900)
    global fls_y1
    y1 = copy.deepcopy(commit_model)

    # calculate x
    x = np.arange(y1.fls[-1].nx) * y1.fls[-1].dx * y1.fls[-1].map_dx

    surface_h = y1.fls[-1].bed_h
    # start array for optimization
    x0 = surface_h[np.linspace(0,hef_fls[-1].nx-1,15).astype(int)]
    #x0 = (y1.fls[-1].bed_h+y1.fls[-1].thick/4)[np.linspace(0,hef_fls[-1].nx-1,25).astype(int)]

    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'ineq', 'fun': con2},
            {'type': 'ineq', 'fun': con3},
            {'type': 'ineq', 'fun': con4},
            {'type': 'ineq', 'fun': con5}
            )

    pool = mp.Pool(processes=4)
    results = [pool.apply_async(parallel, args=(x,x0,cons)) for x in range(75,275,12)]
    output = [p.get() for p in results]
    for index,shape in enumerate(output):
        try:
            end_model = run_model(shape)
            ax[0].plot(x,end_model.fls[-1].surface_h,alpha=0.5)
            end_model.run_until(1900)
            ax[1].plot(x, end_model.fls[-1].surface_h,alpha=0.5)
        except:
            pass
    ax[0].plot(x,fls_y0[-1].bed_h ,'k',label='bed')
    ax[0].plot(x, fls_y0[-1].surface_h, 'k', label='solution')
    ax[0].set_ylabel('Altitude (m)')
    ax[0].set_xlabel('Distance along the flowline (m)')
    ax[0].set_title('1850')
    ax[0].legend(loc='best')

    ax[1].plot(x, fls_y0[-1].bed_h, 'k', label='bed')
    ax[1].plot(x, y1.fls[-1].surface_h, 'k', label='solution')
    ax[1].legend(loc='best')
    ax[1].set_ylabel('Altitude (m)')
    ax[1].set_xlabel('Distance along the flowline (m)')
    ax[1].set_title('1900')
    plt.show()

    '''
    #res = minimize(objfunc, x0, method='COBYLA', tol=1e-04, constraints=cons,
    #               options={'maxiter': 5000, 'rhobeg': 200})

    end_model = run_model(res.x)

    print(res)

    end_model = run_model(res.x)
    ax1.plot(fls_y0[-1].surface_h, 'k')


    ax3.plot(x,end_model.fls[-1].bed_h ,'k')
    ax3.plot(x,end_model.fls[-1].surface_h,color='teal',label='res.x 2')
    ax3.plot(x[np.linspace(0,y1.fls[-1].nx-1,len(res.x)).astype(int)], end_model.fls[-1].surface_h[np.linspace(0,y1.fls[-1].nx-1,len(res.x)).astype(int)],'o',color='teal')
    ax3.plot(x, end_model.fls[-1].surface_h, label='optimized y0')
    ax3.plot(x,fls_y0[-1].surface_h,label='real y0')
    end_model.run_until(1900)

    ax3.plot(x,end_model.fls[-1].surface_h,label='optimized y1')
    ax3.plot(x, y1.fls[-1].surface_h, label=' real y1')
    ax3.legend(loc='best')

    plt.show()
    '''
