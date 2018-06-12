from scipy.optimize import minimize
# Scientific packages
import numpy as np
# Constants
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.massbalance import LinearMassBalance
from oggm.core.flowline import FluxBasedModel
from oggm.core.flowline import RectangularBedFlowline
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


def min_length_m(fls):
    thick = fls[-1].surface_h-fls[-1].bed_h
    # We define the length a bit differently: but more robust
    pok = np.where(thick == 0.)[0]
    return pok[0]* fls[-1].dx_meter

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
    init_flowline = RectangularBedFlowline(surface_h=rescale(surface_h,nx), bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalance(3000, grad=4)
    #annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR

    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    model.run_until(300)

    measured = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))
    f = abs(model.fls[-1].surface_h - measured.fls[-1].surface_h)+\
        abs(min_length_m(model.fls)-measured.length_m)+\
        abs(model.area_km2-measured.area_km2)+\
        abs(model.volume_km3-measured.volume_km3)
    print(sum(f))
    return sum(f)


def con1(surface_h):
    bed_h = np.linspace(3400,1400,len(surface_h))
    return surface_h-bed_h


def con2(surface_h):
    bed_h = np.linspace(3400, 1400, len(surface_h))
    return bed_h+1000-surface_h


def con3(surface_h):
    bed_h = np.linspace(3400, 1400, len(surface_h))
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


def run_model(surface_h):
    nx = 200
    bed_h = np.linspace(3400, 1400, nx)
    # At the begining, there is no glacier so our glacier surface is at the bed altitude

    # Let's set the model grid spacing to 100m (needed later)
    map_dx = 100

    # The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
    widths = np.zeros(nx) + 3.
    # Define our bed
    init_flowline = RectangularBedFlowline(surface_h=rescale(surface_h, nx),
                                         bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalance(3000, grad=4)
#    annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR

    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    return model


if __name__ == '__main__':
    #v_bed_h=np.vectorize(bed_h)

    #x0 = np.linspace(3400, 1400,15)

    #x0=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_200.pkl','rb')).fls[-1].surface_h[np.linspace(0,199,30).astype(int)]
    x0 = rescale(pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/result_15pt_noCon5.txt','rb')),200)
    #print(x_coord)
    cons = ({'type': 'ineq', 'fun': con1},
            {'type': 'ineq', 'fun': con2},
            {'type': 'ineq', 'fun': con3},
            {'type': 'ineq', 'fun': con4}
            #{'type': 'ineq', 'fun': con5}
            )

    #res = minimize(objfunc, x0,method='COBYLA',tol=1e-04,constraints=cons,options={'maxiter':5000,'rhobeg' :50})
    #res = minimize(objfunc, x0, method='Nelder-Mead', tol=1e-04, options={'maxiter': 5000})


    import pickle
    #pickle._dump(res.x,open('result_15pt_noCon5.txt','wb'))
    #result=res.x
    result=pickle.load(open('result_200pt.txt','rb'))

    start_model= run_model(result)
    print(objfunc(result))
    print(start_model.length_m, start_model.area_m2,start_model.volume_m3)
    #print(start_model.yr,start_model.length_m/start_model.fls[-1].dx_meter)

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(np.linspace(3400,1400,200),'k',label='bed',linewidth=1)
    axarr[0].plot(pickle.load(
        open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl',
             'rb')).fls[-1].surface_h, color='teal',
             label='"real" initial state')
    axarr[0].plot(start_model.fls[-1].surface_h, color='tomato',label='optimized initial state')
    axarr[0].plot(np.linspace(0,200,200),result, 'o',color = 'tomato')
    axarr[0].legend(loc='best')
    axarr[0].set_ylabel('Altitude (m)')
    axarr[0].set_title('t=150')

    start_model.run_until(300)
    axarr[1].plot(np.linspace(3400, 1400, 200), 'k', label='bed', linewidth=1)
    axarr[1].plot(pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb')).fls[-1].surface_h, color='teal',label='"real" final state')
    axarr[1].plot(start_model.fls[-1].surface_h,color='tomato', label='optimized initial state')
    axarr[1].set_title('t=300')
    #plt.plot(np.linspace(0,200,200),result,'o',color='teal')
    #plt.plot(pickle.load(
    #    open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl',
    #         'rb')).fls[-1].surface_h, color='aquamarine',label='"real" initial state')

    #oggm_length=int(start_model.fls[-1].length_m / start_model.fls[-1].dx_meter)



    start_model.run_until(300)
    #plt.plot(start_model.fls[-1].surface_h, color='tomato',label='opitmized t=300')
    #plt.plot(pickle.load(
    #    open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl',
    #         'rb')).fls[-1].surface_h,  color='teal',
    #         label='"real" final state')

    plt.legend(loc='best')
    plt.xlabel('Grid Points')
    plt.ylabel('Altitude (m)')

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