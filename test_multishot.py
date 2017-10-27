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
import math
import multiprocessing as mp
import copy

def min_length_m(fls):
    thick = fls[-1].surface_h-fls[-1].bed_h
    # We define the length a bit differently: but more robust
    pok = np.where(thick == 0.)[0]
    try:
        return pok[0] * fls[-1].dx_meter
    except:
        return fls[-1].length_m


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

def obj(array):
    array=[array[:int((len(array)/2))],array[int(len(array)/2):]]
    pool = mp.Pool(processes=4)
    results = pool.map(run_model, array)
    pool.close()
    pool.join()
    try:
        results[0].run_until(175)
        results[1].reset_y0(175)
        results1=copy.deepcopy(results[1])
        results[1].run_until(300)
        #print(results[0].yr,results1.yr,results[1].yr)
    except:
        return np.nan
    #minimize difference between measurements
    f1=sum(abs(results[1].fls[-1].surface_h - final_flowline.fls[-1].surface_h)) + \
       abs(results[1].area_km2 - final_flowline.area_km2)+abs(results[1].volume_km3 - final_flowline.volume_km3)+\
       abs(min_length_m(results[1].fls) - min_length_m(final_flowline.fls))

    #minimize difference between shots
    f2=sum(abs(results[0].fls[-1].surface_h-results1.fls[-1].surface_h))+\
       abs(results1.area_km2 - results[0].area_km2)+abs(results1.volume_km3 - results[0].volume_km3)+\
       abs(min_length_m(results1.fls) - min_length_m(results[0].fls))

    print(f1+f2)
    return(f1+f2)


def con1(array):
    array = [array[:int((len(array) / 2))], array[int(len(array) / 2):]]
    bed_h = np.linspace(3400,1400,len(array[0]))
    return np.concatenate((array[0]-bed_h,array[1]-bed_h))


def con2(array):
    array = [array[:int((len(array) / 2))], array[int(len(array) / 2):]]
    bed_h = np.linspace(3400, 1400, len(array[0]))
    return np.concatenate((bed_h+1000-array[0],bed_h+1000-array[1]))


def con3(array):
    array = [array[:int((len(array) / 2))], array[int(len(array) / 2):]]
    bed_h = np.linspace(3400, 1400, len(array[0]))
    return np.sum((bed_h[-1]-array[0][-1],bed_h[-1]-array[1][-1]))


def con4(array):
    array = [array[:int((len(array) / 2))], array[int(len(array) / 2):]]
    con=[]
    try:
        for surface_h in array:
            h = rescale(surface_h,200)
            bed_h = np.linspace(3400, 1400, 200)
            bed_h_20 = np.linspace(3400,1400,len(surface_h))
            length = np.where(h - bed_h < 5)[0][0]
            coord = np.linspace(0,200,len(surface_h))
            length_index = np.where(coord>length)[0][0]
            np.concatenate((con,bed_h_20[length_index::]-surface_h[length_index::]))
    except:
        return np.nan
    return con


def con5(array):
    array = [array[:int((len(array) / 2))], array[int(len(array) / 2):]]
    return 10-(abs(final_flowline.fls[-1].surface_h[0]-array[0][0]))


def glacier_length(surface_h):
    bed_h = np.linspace(3400, 1400, 200)
    h = rescale(surface_h, 200)
    length = np.where(h - bed_h < 5)[0][0]
    coord = np.linspace(0, 200, len(surface_h))
    length_index = np.where(coord > length)[0][0]
    return [length_index,length]

if __name__ == '__main__':
    global final_flowline
    final_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))
    initial_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))
    #x0 = np.concatenate((rescale(final_flowline.fls[-1].surface_h,15),rescale(final_flowline.fls[-1].surface_h,15)))
    x0 = np.concatenate((np.linspace(3400, 1400, 15),np.linspace(3400,1400,15)))
    #obj(x0)
    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'ineq', 'fun': con2},
            {'type': 'ineq', 'fun': con3},
            {'type': 'ineq', 'fun': con4},
            {'type': 'ineq', 'fun': con5},
            )
    res = minimize(obj, x0, method='COBYLA', tol=1e-04, constraints=cons,options={'maxiter': 5000, 'rhobeg': 150})

    #pickle.dump(res,open('result_multishot.txt', 'wb'))
    #mb_model = LinearMassBalanceModel(3000, grad=4)
    #res = pickle.load(open('result_multishot.txt', 'rb'))
    opt_model = run_model(res.x[:int(len(res.x)/2)])
    #print(glacier_length(opt_model.fls[-1].surface_h),opt_model.length_m/opt_model.fls[-1].dx_meter)
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(np.linspace(3400, 1400, 200),'k',label='bedrock')
    axarr[0].plot(initial_flowline.fls[-1].surface_h,color='teal', label='"real" inital state')
    axarr[0].plot(np.linspace(0,200,15),res.x[:int(len(res.x)/2)], 'o',color='gold',)
    axarr[0].plot(opt_model.fls[-1].surface_h, color='gold', label='optimized initial state 150')
    axarr[0].set_ylabel('Altitude (m)')
    axarr[0].set_title('t=150')
    axarr[0].legend(loc='best')

    initial_flowline.run_until(175)
    opt_model.run_until(175)

    axarr[1].plot(np.linspace(3400, 1400, 200),'k',label='bedrock')
    axarr[1].plot(initial_flowline.fls[-1].surface_h, color='teal',label='"real" state')
    axarr[1].plot(opt_model.fls[-1].surface_h, color='gold', label= 'optimized state 150')
    axarr[1].plot(np.linspace(0,200,15),res.x[int(len(res.x)/2):],'o',color='tomato')
    axarr[1].plot(np.linspace(0, 200, 15), res.x[int(len(res.x) / 2):],color='tomato', label='optimized state 175')
    axarr[1].set_ylabel('Altitude (m)')
    axarr[1].set_title('t=175')
    axarr[1].legend(loc='best')
    initial_flowline.run_until(300)
    opt_model2= run_model(res.x[int(len(res.x)/2):])
    opt_model2.reset_y0(175)
    opt_model2.run_until(300)

    axarr[2].plot(np.linspace(3400, 1400, 200),'k',label='bedrock')
    axarr[2].plot(initial_flowline.fls[-1].surface_h, color='teal',label='"real" state')
    axarr[2].plot(opt_model2.fls[-1].surface_h, color='tomato', label= 'optimized state 175')
    axarr[2].set_ylabel('Altitude (m)')
    axarr[2].set_title('t=300')
    axarr[2].legend(loc='best')

    plt.show()
