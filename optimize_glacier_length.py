from scipy.optimize import minimize
# Scientific packages
import numpy as np
# Constants
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.models.massbalance import LinearMassBalanceModel
from oggm.core.models.flowline import FluxBasedModel
from oggm.core.models.flowline import VerticalWallFlowline, \
    TrapezoidalFlowline, ParabolicFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import scipy.interpolate as sp
import pickle
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

FlowlineModel = partial(FluxBasedModel, inplace=False)


def bed(points):
    bed=[]
    for i in points:
        bed.append(3400-10*i)
    return bed

def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    #old_indices = np.arange(0, len(array))
    #new_length = mx
    #new_indices = np.linspace(0, len(array)-1, new_length)
    grid_x0 = np.linspace(0,mx,10,endpoint=True)

    fc = sp.interp1d(grid_x0, array, kind='cubic')

    return fc(np.linspace(0,mx,mx,endpoint=True))
    #spl = UnivariateSpline(old_indices, array, k=1, s=0)
    #new_array = spl(new_indices)
    #print(new_array)
    #return new_array


def make_surface_h(x):

    length = int(round(x[0]))
    glacier = rescale(x[1:], length)
    surface_h = np.insert(glacier, len(glacier), bed_h[-(200 - length):])
    return surface_h

def objfunc(x):

    nx = 200
    # set the model grid spacing to 100m
    map_dx = 100
    # The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
    widths = np.zeros(nx) + 3.

    surface_h = make_surface_h(x)
    flowline = VerticalWallFlowline(surface_h=surface_h, bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    #plt.plot(flowline.surface_h)
    #plt.plot(np.linspace(0,x[0],10),x[1::],'o')
    #plt.plot(bed_h)
    #plt.show()
    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalanceModel(3000, grad=4)

    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(flowline, mb_model=mb_model, y0=150)
    model.run_until(300)
    #plt.plot(model.fls[-1].surface_h)
    measured = pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl',
             'rb'))
    #plt.plot(measured.fls[-1].surface_h)
    #plt.show()

    f = abs(model.fls[-1].surface_h - measured.fls[-1].surface_h)+\
        abs(model.length_m-measured.length_m)+\
        abs(model.area_km2-measured.area_km2)+\
        abs(model.volume_km3-measured.volume_km3)
    print('objective:',sum(f))
    return f


def con1(x):
    x_grid = np.linspace(0,x[0],10,endpoint=True)
    #bed_h = np.linspace(3400,1400,len(x[1:]))
    return x[1:]-bed(x_grid)

def con2(x):
    x_grid = np.linspace(0, x[0], 10, endpoint=True)

    return bed(x_grid)-x[1:]+1000

def con3(x):
    x_grid = np.linspace(0, x[0], 10, endpoint=True)
    return bed(x_grid)-x[1:]

def con4(x):
    x_grid = np.linspace(0, x[0], 10, endpoint=True)
    return bed(x_grid)[-1]-x[1:][-1]

def con5(x):
    return 100-x[0]

if __name__ == '__main__':
    global grid
    global bed_h
    grid = np.linspace(0, 200, 200, endpoint=True)
    bed_h = bed(grid)
    measured = pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl',
             'rb')).fls[-1].surface_h
    fl = sp.interp1d(grid, measured, kind='cubic')
    length = 55
    print(fl(np.linspace(0,length,10, endpoint=True)))
    x0 = np.insert([length],1,fl(np.linspace(0,length,10, endpoint=True)))
    print(x0)
    objfunc(x0)

    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'ineq', 'fun': con2},
            {'type': 'ineq', 'fun': con3},
            {'type': 'ineq', 'fun': con4},
            {'type': 'ineq', 'fun': con5}
            )
    res = minimize(objfunc, x0,method='COBYLA',tol=1e-04,constraints=cons,options={'maxiter':5000,'rhobeg' :100})
    print(res)
    print(res.x)
    plt.figure(1)
    plt.plot(measured)
    plt.plot(grid,make_surface_h(res.x))
    plt.plot(np.linspace(0,res.x[0],10,endpoint=True),res.x[1::],'o')
    plt.show()