from scipy.optimize import differential_evolution
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
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
FlowlineModel = partial(FluxBasedModel, inplace=False)

from pyswarm import pso

def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    old_indices = np.arange(0, len(array))
    new_length = mx
    new_indices = np.linspace(0, len(array) - 1, new_length)
    spl = UnivariateSpline(old_indices, array, k=1, s=0)
    new_array = spl(new_indices)
    return new_array

def objfunc(surface_h):


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
    init_flowline = VerticalWallFlowline(surface_h=rescale(surface_h,nx), bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalanceModel(3000, grad=4)
    annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR

    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    model.run_until(300)

    measured = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))
    f = abs(model.fls[-1].surface_h - measured.fls[-1].surface_h)+\
        abs(model.length_m-measured.length_m)+\
        abs(model.area_km2-measured.area_km2)+\
        abs(model.volume_km3-measured.volume_km3)
    print(sum(f))
    return sum(f)

def con4(surface_h):
    h = rescale(surface_h,200)
    bed_h = np.linspace(3400, 1400, 200)
    bed_h_20 = np.linspace(3400,1400,len(surface_h))
    length = np.where(h - bed_h < 1)[0][0]
    coord = np.linspace(0,200,len(surface_h))
    length_index = np.where(coord>length)[0][0]
    return sum(bed_h_20[length_index::]-surface_h[length_index::])


if __name__ == '__main__':
    lb = [-3, -1]
    ub = [2, 6]
    bed_h = np.linspace(3400, 1400, 10)
    upper_bounds = bed_h+500

    xopt, fopt = pso(objfunc,bed_h,upper_bounds)
    #xopt, fopt = pso(banana, lb, ub, f_ieqcons=con)

    print(xopt, fopt)
    plt.plot(np.linspace(0,200,10),xopt )
    plt.plot(np.linspace(0, 200, 10), xopt,'o')
    plt.plot(np.linspace(0,200,10),bed_h)
    plt.show()