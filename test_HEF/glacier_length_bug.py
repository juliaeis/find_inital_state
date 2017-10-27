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

measured = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb')).fls[-1].surface_h

ar= [186.76426107,185.94878738,185.05172836,184.06216873,182.96254819,181.73772465,
180.36605834,178.82846391,177.09924633,175.15626245,172.97365307,170.52512883,
167.78289209,164.71752145,161.29680665,157.48370195,153.23283817,148.48483316,
143.15699803,137.12785188,130.210028,122.09857231,112.25789271,99.61874157,
81.43258339,39.48892681]

rev=np.concatenate((ar[::-1],ar[1:]))
measured [120:171]=rev+np.linspace(3400,1400,200)[120:171]


def min_length_m(fls):
    thick = fls[-1].surface_h-fls[-1].bed_h
    # We define the length a bit differently: but more robust
    pok = np.where(thick == 0.)[0]
    return pok[0] * fls[-1].dx_meter
def max_length_m(fls):
    thick = fls[-1].surface_h-fls[-1].bed_h
    # We define the length a bit differently: but more robust
    pok = np.where(thick > 0.)[0]
    return (pok[-1]+1) * fls[-1].dx_meter

def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    old_indices = np.arange(0, len(array))
    new_length = mx
    new_indices = np.linspace(0, len(array) - 1, new_length)
    spl = UnivariateSpline(old_indices, array, k=1, s=0)
    new_array = spl(new_indices)
    return new_array

def run_model(surface_h):
    nx = 200
    bed_h = np.linspace(3400, 1400, nx)
    # At the begining, there is no glacier so our glacier surface is at the bed altitude

    # Let's set the model grid spacing to 100m (needed later)
    map_dx = 100

    # The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
    widths = np.zeros(nx) + 3.
    # Define our bed
    init_flowline = VerticalWallFlowline(surface_h=rescale(surface_h, nx),
                                         bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalanceModel(3000, grad=4)
    annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR

    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    return model

model=run_model(measured)
model.run_until(160)
x=np.arange(model.fls[-1].nx) * model.fls[-1].dx * model.fls[-1].map_dx

plt.figure(1)
plt.plot(x,np.linspace(3400,1400,200),'k',label='bed')

plt.plot(x,model.fls[-1].surface_h,color='teal',label='surface_h')
plt.plot(model.length_m,model.fls[-1].bed_h[int(np.where(x==model.length_m)[0])],'o',color='teal',label='oggm glacier length (m)')
plt.plot(min_length_m(model.fls),model.fls[-1].bed_h[int(np.where(x==min_length_m(model.fls))[0])],'o', color='lightgreen', label ='glacier minimum length (m)')
plt.plot(max_length_m(model.fls),model.fls[-1].bed_h[int(np.where(x==max_length_m(model.fls))[0])],'o', color='tomato', label ='glacier maximum length (m)')
plt.legend(loc='best')
plt.xlabel('Distance along the flowline (m)')
plt.ylabel('Altitude (m)')
plt.show()
