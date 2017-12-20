from oggm import cfg, workflow, tasks
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import LinearMassBalance, PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
from functools import partial
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


def run_model(param,gdir,ela):
    climate = LinearMassBalance(ela_h=ela)
    climate.temp_bias=param
    fls = gdir.read_pickle('model_flowlines')
    fls1 = copy.deepcopy(fls)
    fls1[-1].surface_h = y_start.fls[-1].surface_h
    model = FluxBasedModel(fls1, mb_model=climate, glen_a=cfg.A, y0=0)
    model.run_until(50)

    fls2 = copy.deepcopy(fls)
    fls2[-1].surface_h = copy.deepcopy(model.fls[-1].surface_h)
    real_model = FluxBasedModel(fls2, mb_model=past_climate,
                                glen_a=cfg.A, y0=1850)
    real_model.run_until(1900)
    return [model,real_model]

def objfunc(param,gdir,ela):

    try:
        model, real_model = run_model(param,gdir,ela)
        f = np.sum(abs(real_model.fls[-1].surface_h - y_1900.fls[-1].surface_h))**2 + \
                abs(real_model.length_m - y_1900.length_m)**2 + \
                abs(real_model.area_m2 - y_1900.area_m2) + \
                abs(real_model.volume_m3-y_1900.volume_m3)
    except:
        f=np.inf
    #print(param,f)
    return f

def find_initial_state(gdir):

    global past_climate
    global y_1900
    global y_start

    fls = gdir.read_pickle('model_flowlines')
    past_climate = PastMassBalance(gdir)
    commit_model = FluxBasedModel(fls, mb_model=past_climate,
                                  glen_a=cfg.A, y0=1850)
    y_1850 = copy.deepcopy(commit_model)
    commit_model.run_until(1900)
    y_1900 = copy.deepcopy(commit_model)
    x = np.arange(y_1900.fls[-1].nx) * y_1900.fls[-1].dx * y_1900.fls[-1].map_dx

    plt.figure()

    ax1 = plt.subplot(311)
    ax1.set_title(gdir.rgi_id)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot(x, y_1850.fls[-1].surface_h, 'k:', label='solution')
    plt.plot(x, y_1850.fls[-1].bed_h, 'k', label='bed')
    plt.legend(loc='best')

    ax2 = plt.subplot(312, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.plot(x, y_1900.fls[-1].surface_h, 'k:', label='solution')
    ax2.plot(x, y_1900.fls[-1].bed_h, 'k', label='bed')

    ax3 = plt.subplot(313,sharex=ax1)
    ax3.plot(x, np.zeros(len(x)), 'k--')

    growing_climate = LinearMassBalance(past_climate.get_ela(1850),3)

    growing_model = FluxBasedModel(fls, mb_model=growing_climate,
                                  glen_a=cfg.A, y0=1850)
    #growing_model.fls[-1].surface_h=growing_model.fls[-1].bed_h
    succes = 0



    y_start = copy.deepcopy(growing_model)
    y_start.run_until(1950)

    res = minimize(objfunc, 0.5,args=(gdir,past_climate.get_ela(1850),), method='COBYLA',
                   tol=1e-04, options={'maxiter':500,'rhobeg':5})
    #print(res)

    result_model_1850,result_model_1900 = run_model(res.x,gdir,past_climate.get_ela(1850))
    dif = result_model_1900.fls[-1].surface_h-y_1900.fls[-1].surface_h
    s = np.sum(np.abs(dif))
    print(gdir.rgi_id,s)
    if s<25:
        #print(gdir.rgi_id, i)
        succes +=1
        ax1.plot(x, result_model_1850.fls[-1].surface_h, label='optimum')
        ax2.plot(x, result_model_1900.fls[-1].surface_h, label='optimum')
        ax3.plot(x, dif)

    '''
    #ax[0].plot(x, y_start.fls[-1].surface_h, label=y_start.yr)
    ax[0].plot(x,result_model_1850.fls[-1].surface_h, label = 'optimum')
    ax[1].plot(x, result_model_1900.fls[-1].surface_h,
               label='optimum')

    ax[0].plot(x, y_1850.fls[-1].surface_h,':', label=y_1850.yr)
    ax[0].plot(x, y_1850.fls[-1].bed_h, 'k--')
    ax[0].legend(loc='best')
    ax[0].set_title(gdir.rgi_id)

    ax[1].plot(x,y_1900.fls[-1].surface_h,':',label = y_1900.yr)
    ax[1].plot(x, y_1900.fls[-1].bed_h, 'k--')
    ax[1].legend(loc='best')
    '''
    ax1.legend(loc='best')
    if succes>0:
        plot_dir = os.path.join(cfg.PATHS['working_dir'],'plots','surface_h')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir,gdir.rgi_id+'.png'))
        plt.show()
    return True

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
    pool = mp.Pool(processes=4)
    pool.map(find_initial_state,gdirs)




