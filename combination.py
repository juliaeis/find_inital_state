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
import pickle
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

FlowlineModel = partial(FluxBasedModel, inplace=False)
global measured
measured = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))
measured150 = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))

def find_guess(mb_data):

    nx = 200
    bed_h = np.linspace(3400, 1400, nx)
    map_dx = 100
    widths = np.zeros(nx) + 3.

    init_flowline = VerticalWallFlowline(surface_h=measured.fls[-1].surface_h, bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    # with random ELA
    mb_model = LinearMassBalanceModel(mb_data[0], grad=4)
    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    model.run_until(300)

    mb_model_orig=LinearMassBalanceModel(3000,grad=4)
    end_model= FlowlineModel(model.fls, mb_model=mb_model_orig, y0=150)
    end_model.run_until(300)

    f = abs(end_model.fls[-1].surface_h - measured.fls[-1].surface_h)+\
        abs(end_model.length_m-measured.length_m)+\
        abs(end_model.area_km2-measured.area_km2)+\
        abs(end_model.volume_km3-measured.volume_km3)
    print(sum(f))
    return f

def con1(mb_data):
    return mb_data -2000

def con2(mb_data):
    return 5000-mb_data

#--------------------------------------find starting value------------------
x0 = 3000
cons = ({'type': 'ineq', 'fun': con1},
        {'type': 'ineq', 'fun': con2})

res = minimize(find_guess, x0,method='COBYLA',tol=1e-3,constraints=cons,options={'maxiter':5000,'rhobeg' :400})

nx = 200
bed_h = np.linspace(3400, 1400, nx)
# At the begining, there is no glacier so our glacier surface is at the bed altitude

# Let's set the model grid spacing to 100m (needed later)
map_dx = 100

# The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
widths = np.zeros(nx) + 3.
mb_model = LinearMassBalanceModel(res.x, grad=4)
init_flowline = VerticalWallFlowline(surface_h=measured.fls[-1].surface_h, bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
# The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
model.run_until(300)

start_guess=model.fls[-1].surface_h
print('####### beginn optimization######')

#------------------------------optimization---------------------------------
def objfunc(surface_h):
    nx = 200
    bed_h = np.linspace(3400, 1400, nx)
    # At the begining, there is no glacier so our glacier surface is at the bed altitude

    # Let's set the model grid spacing to 100m (needed later)
    map_dx = 100

    # The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
    widths = np.zeros(nx) + 3.
    # Define our bed
    init_flowline = VerticalWallFlowline(surface_h=surface_h,
                                         bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    mb_model = LinearMassBalanceModel(3000, grad=4)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    model.run_until(300)

    f = abs(model.fls[-1].surface_h - measured.fls[-1].surface_h)+\
        abs(model.length_m-measured.length_m)+\
        abs(model.area_km2-measured.area_km2)+\
        abs(model.volume_km3-measured.volume_km3)
    print(sum(f))
    return f

def con3(surface_h):
    bed_h = np.linspace(3400,1400,len(surface_h))
    return surface_h-bed_h


def con4(surface_h):
    bed_h = np.linspace(3400, 1400, len(surface_h))
    return bed_h+1000-surface_h


def con5(surface_h):
    bed_h = np.linspace(3400, 1400, len(surface_h))
    return bed_h[-1]-surface_h[-1]

def con6(surface_h):
    return 10-(abs(measured.fls[-1].surface_h[0]-surface_h[0]))

cons = ({'type': 'ineq', 'fun': con3},
        {'type': 'ineq', 'fun': con4},
        {'type': 'ineq', 'fun': con5},
        {'type': 'ineq', 'fun': con6},
        )
res = minimize(objfunc, start_guess,method='COBYLA',tol=1e-04,constraints=cons,options={'maxiter':5000,'rhobeg' :50})

plt.figure()
plt.plot(bed_h, color='k', label='Bedrock')
plt.plot(measured.fls[-1].surface_h, label='final glacier')
plt.plot(measured150.fls[-1].surface_h, label='original intial glacier')
plt.plot(model.fls[-1].surface_h, label='start guess')
plt.plot(res.x,label='optimized initial glacier')
#plt.plot(end_model.fls[-1].surface_h, label= 'optimized final glacier')
plt.xlabel('Grid points')
plt.ylabel('Altitude (m)')
plt.legend(loc='best');
plt.show()
