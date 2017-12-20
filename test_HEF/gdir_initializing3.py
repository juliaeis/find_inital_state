from oggm import cfg, workflow, tasks
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import LinearMassBalance, PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
from functools import partial
FlowlineModel = partial(FluxBasedModel, inplace=False)
from bayes_opt import BayesianOptimization

import os
import salem
import copy
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import multiprocessing as mp
import time


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


def run_model(param,gdir,start_fls):
    fls = gdir.read_pickle('model_flowlines')
    fls1 = copy.deepcopy(fls)
    fls1[-1].surface_h = copy.deepcopy(start_fls[-1].surface_h)
    climate= copy.deepcopy(past_climate)
    climate.temp_bias = param
    model = FluxBasedModel(fls1, mb_model=climate,
                           glen_a=cfg.A, y0=1850)
    model.run_until(1900)
    fls2= copy.deepcopy(fls)
    fls2[-1].surface_h=model.fls[-1].surface_h
    real_model = FluxBasedModel(fls2, mb_model=past_climate,
                                glen_a=cfg.A, y0=1850)

    real_model.run_until(1900)
    return [model,real_model]

def objfunc(param,gdir,start_fls):

    try:
        model, real_model = run_model(param,gdir,start_fls)
        f = np.sum(abs(real_model.fls[-1].surface_h - y_1900.fls[-1].surface_h))**2 + \
                abs(real_model.length_m - y_1900.length_m)**2 + \
                abs(real_model.area_m2 - y_1900.area_m2) + \
                abs(real_model.volume_m3-y_1900.volume_m3)
    except:
        f=np.inf
    print(param, f)
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



    #y_start = copy.deepcopy(growing_model)
    #y_start.run_until(1950)
    '''
    res = minimize(objfunc, [0.5,1900],args=(gdir,y_1900.fls,), method='COBYLA',
                   tol=1e-04, options={'maxiter':500,'rhobeg':2})
    #print(res)
    '''
    res= differential_evolution (objfunc,bounds=[(-5,5)],args=(gdir,y_1900.fls),popsize=50)

    result_model_1850,result_model_1900 = run_model(res.x,gdir,y_1900.fls)

    dif = result_model_1900.fls[-1].surface_h-y_1900.fls[-1].surface_h
    s = np.sum(np.abs(dif))

    ax1.plot(x, result_model_1850.fls[-1].surface_h, label='optimum 1')
    ax2.plot(x, result_model_1900.fls[-1].surface_h)
    ax3.plot(x, dif)

    ax1.legend(loc='best')

    plot_dir = os.path.join(cfg.PATHS['working_dir'],'plots','differential_evolution_popsize_30')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir,gdir.rgi_id+'.png'))
    plt.show()
    #return True

if __name__ == '__main__':
    start_time = time.time()
    cfg.initialize()

    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
    #cfg.PATHS['working_dir'] = '/home/juliaeis/PycharmProjects/find_inital_state/test_HEF'
    cfg.PATHS['working_dir'] = os.environ.get("S_WORKDIR")
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['prcp_scaling_factor']
    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['optimize_inversion_params'] = True

    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

    rgi = get_demo_file('rgi_oetztal.shp')
    gdirs = workflow.init_glacier_regions(salem.read_shapefile(rgi))
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)


    prepare_for_initializing(gdirs)
    pool = mp.Pool()
    pool.map(find_initial_state,gdirs)
    '''
    for gdir in gdirs:
        if gdir.rgi_id == "RGI50-11.00719_d01":
            find_initial_state(gdir)
    '''
    print(time.time()-start_time)
