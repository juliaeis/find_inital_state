from oggm import cfg, workflow, tasks
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import LinearMassBalance, PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
from functools import partial
from scipy.interpolate import UnivariateSpline
FlowlineModel = partial(FluxBasedModel, inplace=False)

import os
import salem
import copy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp


def prepare_for_initializing(gdirs):

    list_tasks = [
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.catchment_area,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape
    ]
    for task in list_tasks:
        workflow.execute_entity_task(task, gdirs)

    workflow.climate_tasks(gdirs)
    workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs)
    glen_a = cfg.A

    for gdir in gdirs:
        mass_conservation_inversion(gdir, glen_a=glen_a)

    workflow.execute_entity_task(tasks.optimize_inversion_params,gdirs)
    workflow.execute_entity_task(tasks.volume_inversion, gdirs)
    workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
    workflow.execute_entity_task(tasks.init_present_time_glacier,gdirs)


def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    old_indices = np.arange(0, len(array))
    new_length = mx
    new_indices = np.linspace(0, len(array) - 1, new_length)
    spl = UnivariateSpline(old_indices, array, k=1, s=0)
    new_array = spl(new_indices)
    return new_array


def run_model(param,gdir):
    nx = y_1900.fls[-1].nx

    fls = gdir.read_pickle('model_flowlines')
    surface_h = rescale(param,nx)

    thick = surface_h-y_1900.fls[-1].bed_h
    # We define the length a bit differently: but more robust
    try:
        pok = np.where(thick < 10)[0]
        surface_h[int(pok[0]):] = y_1900.fls[-1].bed_h[int(pok[0]):]
        fls[-1].surface_h = surface_h
        real_model = FluxBasedModel(fls, mb_model=past_climate, glen_a=cfg.A,
                                    y0=1850)
        model = copy.deepcopy(real_model)
        real_model.run_until(1900)

        return [model, real_model]

    except:
        pass




def objfunc(param,gdir):

    try:
        model, real_model = run_model(param,gdir)
        f = np.sum(abs(real_model.fls[-1].surface_h - y_1900.fls[-1].surface_h))**2 #+ \
                #abs(real_model.length_m - y_1900.length_m)**2 + \
                #abs(real_model.area_m2 - y_1900.area_m2) ** 2 + \
                #abs(real_model.volume_m3-y_1900.volume_m3)
    except:
        f=np.inf
    print(f)
    return f

def con1(surface_h,gdir):
    ''' ice thickness greater than zero'''
    try:
        model, real_model =  run_model(surface_h, gdir)
        return model.fls[-1].thick[np.linspace(0,y_1900.fls[-1].nx-1,len(surface_h)).astype(int)]
    except:
        return -np.inf


def con2(surface_h, gdir):
    ''' ice thickness smaller than 1000'''
    try:
        model, real_model = run_model(surface_h, gdir)
        return -model.fls[-1].thick[np.linspace(0,y_1900.fls[-1].nx-1,len(surface_h)).astype(int)]+1000
    except:
        return -np.inf


def con3(surface_h, gdir):
    '''last pixel has to be zero ice thickness'''
    try:
        model, real_model = run_model(surface_h, gdir)
        return -model.fls[-1].thick[-1]
    except:
        return -np.inf
def con4(surface_h, gdir):
    '''glacier change not as much at the head'''
    return 10-(abs(y_1900.fls[-1].surface_h[0]-surface_h[0]))

def con5(surface_h, gdir):
    try:
        model, real_model = run_model(surface_h, gdir)
        surface_h = model.fls[-1].surface_h[np.linspace(0,y_1900.fls[-1].nx-1,len(surface_h)).astype(int)]
        bed_h = model.fls[-1].bed_h[np.linspace(0, y_1900.fls[-1].nx - 1, len(surface_h)).astype(int)]
        con_array = np.zeros(len(surface_h))
        for index in range(1,len(surface_h)):
            if (surface_h[index]!= bed_h[index])and(surface_h[index]> surface_h[index-1]):
                con_array[index]=surface_h[index-1]- surface_h[index]
        return con_array
    except:
        return -np.inf

def parallel(rhobeg,x0,cons,gdir):
    res = minimize(objfunc, x0, args= (gdir,),method='COBYLA', tol=1e-04, constraints=cons,
                   options={'maxiter': 5, 'rhobeg': rhobeg})

    #if res.success:
    return res.x


def find_initial_state(gdir):

    global past_climate
    global y_1900
    global y_start

    f, ax = plt.subplots(2, sharex=True)

    fls = gdir.read_pickle('model_flowlines')
    past_climate = PastMassBalance(gdir)
    commit_model = FluxBasedModel(fls, mb_model=past_climate,
                                  glen_a=cfg.A, y0=1850)
    y_1850 = copy.deepcopy(commit_model)
    commit_model.run_until(1900)
    y_1900 = copy.deepcopy(commit_model)

    x = np.arange(y_1900.fls[-1].nx) * y_1900.fls[-1].dx * y_1900.fls[-1].map_dx
    surface_h = y_1900.fls[-1].bed_h
    x0 = surface_h[np.linspace(0, y_1900.fls[-1].nx - 1, 15).astype(int)]

    cons = ({'type': 'ineq', 'fun': con1,'args': (gdir,)},
            {'type': 'ineq', 'fun': con2,'args': (gdir,)},
            {'type': 'ineq', 'fun': con3,'args': (gdir,)},
            {'type': 'ineq', 'fun': con4,'args': (gdir,)},
            {'type': 'ineq', 'fun': con5,'args': (gdir,)}
            )

    results = [parallel(x, x0, cons,gdir) for x in range(100, 200, 50)]
    #output = [p.get() for p in results]
    output=results
    for index,shape in enumerate(output):
        try:
            model,end_model = run_model(shape, gdir)
            ax[0].plot(x,model.fls[-1].surface_h,alpha=0.5)
            ax[1].plot(x, end_model.fls[-1].surface_h,alpha=0.5)
        except:
            pass
    ax[0].plot(x,y_1900.fls[-1].bed_h ,'k',label='bed')
    ax[0].plot(x, y_1850.fls[-1].surface_h, 'k', label='solution')
    ax[0].set_ylabel('Altitude (m)')
    ax[0].set_xlabel('Distance along the flowline (m)')
    ax[0].set_title('1850')
    ax[0].legend(loc='best')

    ax[1].plot(x, y_1900.fls[-1].bed_h, 'k', label='bed')
    ax[1].plot(x, y_1900.fls[-1].surface_h, 'k', label='solution')
    ax[1].legend(loc='best')
    ax[1].set_ylabel('Altitude (m)')
    ax[1].set_xlabel('Distance along the flowline (m)')
    ax[1].set_title('1900')
    #plt.savefig(os.path.join(cfg.PATHS['working_dir'],'plots','surface_h',gdir.rgi_id+'.png'))
    plt.show()
    print(gdir.rgi_id, 'finished')


if __name__ == '__main__':

    cfg.initialize()

    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
    cfg.PATHS['working_dir'] = '/home/juliaeis/PycharmProjects/find_inital_state/test_HEF'

    cfg.PARAMS['border'] = 80
    cfg.PARAMS['prcp_scaling_factor']
    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['optimize_inversion_params'] = True

    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

    rgi = get_demo_file('rgi_oetztal.shp')
    gdirs = workflow.init_glacier_regions(salem.read_shapefile(rgi))
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    #prepare_for_initializing(gdirs)
    pool = mp.Pool()
    pool.map(find_initial_state,gdirs[:2])



