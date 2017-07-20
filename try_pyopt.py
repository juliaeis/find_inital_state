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

def con1(surface_h):
    bed_h = np.linspace(3400,1400,len(surface_h))
    return surface_h-bed_h

def bed_h(x):
    return 3400-(10*x)

def n_con1(surface_h):

    return surface_h-v_bed_h(new_points)


def con2(surface_h):
    bed_h = np.linspace(3400, 1400, len(surface_h))
    return bed_h+1000-surface_h

def n_con2(surface_h):

    return v_bed_h(new_points)+1000-surface_h


def con3(surface_h):
    bed_h = np.linspace(3400, 1400, len(surface_h))
    return bed_h[-1]-surface_h[-1]
def n_con3(surface_h):
    bed_h = v_bed_h(new_points)
    return bed_h[-1]-surface_h[-1]

def con6(surface_h):
    h = rescale(surface_h, 60)
    bed_h = np.linspace(3400, 1400, 200)[:60]
    #bed_h_20 = np.linspace(3400, 1400, len(surface_h))
    _thick = (rescale(surface_h, 60) - np.linspace(3400, 1400, 200)[:60]).clip(0.)
    pok = np.where(_thick <= 0.)[0]
    length=pok[0]
    #length = np.where(h - bed_h < 1)[0][0]
    coord = np.linspace(0, 60, len(surface_h))
    length_index = np.where(coord > length)[0][0]
    return sum(bed_h[length_index::] - h[length_index::])

def con4(surface_h):
    h = rescale(surface_h,200)
    bed_h = np.linspace(3400, 1400, 200)
    bed_h_20 = np.linspace(3400,1400,len(surface_h))
    length = np.where(h - bed_h < 5)[0][0]
    coord = np.linspace(0,200,len(surface_h))
    length_index = np.where(coord>length)[0][0]
    return sum(bed_h_20[length_index::]-surface_h[length_index::])


def con5(surface_h):
    measured = pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl',
             'rb')).fls[-1].surface_h
    return 10-(abs(measured[0]-surface_h[0]))

def obj2(surface_h):
    #plt.plot(np.arange(0,60+1,5),surface_h,'o')
    bed_h = np.linspace(3400, 1400, 200)[57:]
    surface_h=np.concatenate((rescale(surface_h,60)[:57],bed_h))
    model=run_model(surface_h)
    #plt.plot(model.fls[-1].surface_h)
    model.run_until(300)
    #plt.plot(model.fls[-1].surface_h)
    #plt.show()
    measured = pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl',
             'rb'))
    f = abs(model.fls[-1].surface_h - measured.fls[-1].surface_h) + \
        abs(model.length_m - measured.length_m) + \
        abs(model.area_km2 - measured.area_km2) + \
        abs(model.volume_km3 - measured.volume_km3)
    print(sum(f))
    return sum(f)

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

if __name__ == '__main__':
    v_bed_h=np.vectorize(bed_h)
    import time
    start=time.time()
    x0 = np.linspace(3400, 1400,15)

    #print(x_coord)
    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'ineq', 'fun': con2},
            {'type': 'ineq', 'fun': con3},
            {'type': 'ineq', 'fun': con4},
            {'type': 'ineq', 'fun': con5}
            )

    #res = minimize(objfunc, x0,method='COBYLA',tol=1e-04,constraints=cons,options={'maxiter':5000,'rhobeg' :50})
    #res = minimize(objfunc, x0, method='Nelder-Mead', tol=1e-04, options={'maxiter': 5000})

    print(time.time()-start)
    '''
    plt.figure()
    plt.plot(rescale(res.x,200),'r',label='result')
    plt.plot(np.linspace(0,200,15),res.x, 'ro')
    #plt.plot(np.linspace(0,200,15), res.x, 'r', label='result')
    plt.plot(pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb')).fls[-1].surface_h,label='original')
    plt.legend(loc='best')
    '''
    import pickle
    #pickle._dump(res.x,open('result.txt','wb'))
    #result=res.x
    result=pickle.load(open('result.txt','rb'))

    start_model= run_model(result)
    print(start_model.yr,start_model.length_m/start_model.fls[-1].dx_meter)
    plt.plot(np.linspace(3400,1400,200),'k',label='bed',linewidth=1)
    plt.plot(start_model.fls[-1].surface_h,color='teal', label='optimized initial state')
    plt.plot(np.linspace(0,200,15),result,'o',color='teal')
    plt.plot(pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl',
             'rb')).fls[-1].surface_h,'--', color='teal',label='"real" initial state')

    #oggm_length=int(start_model.fls[-1].length_m / start_model.fls[-1].dx_meter)

    start_model.run_until(300)
    plt.plot(start_model.fls[-1].surface_h, color='tomato',label='opitmized t=300')
    plt.plot(pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl',
             'rb')).fls[-1].surface_h, '--', color='r',
             label='"real" final state')

    plt.legend(loc='best')
    plt.xlabel('Grid Points')
    plt.ylabel('Altitude (m)')

    plt.figure()
    plt.plot(np.linspace(3400, 1400, 200) - rescale(result, 200))
    plt.plot(np.zeros(200), 'k--')

    plt.show()
    '''
    _thick = (rescale(result, 200) - np.linspace(3400, 1400, 200)).clip(0.)
    pok = np.where(_thick <= 0.)[0]
    lenght=pok[0]
    #round length up to nearest tenth
    r_lenght=int(math.ceil(lenght / 10.0)) * 10
    global new_points
    new_points=np.arange(0,r_lenght+1,5)
    solution= pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb')).fls[-1].surface_h[new_points]
    final = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb')).fls[-1].surface_h[new_points]
    new_x0=rescale(result,200)[new_points]
    cons2 = (#{'type': 'ineq', 'fun': n_con1},
            #{'type': 'ineq', 'fun': n_con2},
            #{'type': 'ineq', 'fun': n_con3},
            #{'type': 'ineq', 'fun': con5}
                )
    res = minimize(obj2, solution,method='COBYLA',  tol=1e-03,options={'maxiter':5000,'rhobeg' :50})
    #plt.plot(new_points,new_x0,label='Start value')
    plt.plot(new_points,res.x, label='optimized')
    plt.plot(new_points,res.x,'o',label='Start value')
    plt.plot(pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl',
             'rb')).fls[-1].surface_h[:60], '--', color='teal',
             label='"real" initial state')
    plt.plot(np.linspace(3400, 1400, 200)[:60])
    plt.legend(loc='best')
    plt.show()
    '''