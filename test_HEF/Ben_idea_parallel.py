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
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from matplotlib import pylab
import multiprocessing as mp
import time
import pandas as pd
import pickle


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


def run_model(param,gdir,fls,random_climate2):

    fls1 = copy.deepcopy(fls)
    climate = random_climate2
    climate.temp_bias = param
    model = FluxBasedModel(fls1, mb_model=climate, y0=1890)
    model.run_until_equilibrium()

    fls2= copy.deepcopy(fls)
    #fls2 = model.fls
    for i in range(len(fls)):
        fls2[i].surface_h = model.fls[i].surface_h
    real_model = FluxBasedModel(fls2, mb_model=past_climate,
                                glen_a=cfg.A, y0=1850)

    real_model.run_until(2000)

    return [model,real_model]

def objfunc(param,gdir,fls,random_climate2):

    try:
        model, real_model = run_model(param,gdir,fls,random_climate2)

        f = np.sum(abs(real_model.fls[-1].surface_h - y_2000.fls[-1].surface_h)**2) + \
            np.sum(abs(real_model.fls[-1].widths - y_2000.fls[-1].widths) ** 2) + \
            abs(real_model.length_m - y_2000.length_m)**2
                #abs(real_model.area_m2 - y_1900.area_m2)**2 + \
                #abs(real_model.volume_m3-y_1900.volume_m3)**2

    except:
        f=1e10
    print(param, f)
    return f

def plotxy(ax,x):
    ax.plot(x)


def run_parallel(i,gdir,y_2000):
    try:
        random_climate2 = RandomMassBalance(gdir, y0=1875, halfsize=14)
        res = minimize(objfunc, [0],
                       args=(gdir, y_2000.fls,random_climate2),
                       method='COBYLA',
                       tol=1e-04, options={'maxiter': 100, 'rhobeg': 1})
        # try:
        result_model_1880, result_model_2000 = run_model(res.x, gdir,
                                                         y_2000.fls,random_climate2)
        return result_model_1880,result_model_2000
    except:
        return None, None

def find_initial_state(gdir):

    global past_climate,random_climate2
    global y_2000

    global x
    global ax1,ax2,ax3

    fls = gdir.read_pickle('model_flowlines')
    past_climate = PastMassBalance(gdir)
    random_climate1 = RandomMassBalance(gdir,y0=1865,halfsize=14)


    #construct searched glacier
    commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                  glen_a=cfg.A, y0=1850)
    commit_model.run_until_equilibrium()
    y_1880 = copy.deepcopy(commit_model)

    #construct observed glacier
    commit_model2 = FluxBasedModel(commit_model.fls, mb_model=past_climate,
                                  glen_a=cfg.A, y0=1880)
    commit_model2.run_until(2000)
    y_2000 = copy.deepcopy(commit_model2)

    results  = pd.DataFrame(columns=['1880','2000','length_1880'])


    pool = mp.Pool()
    result_list=pool.map(partial(run_parallel,gdir=gdir,y_2000=y_2000),range(40))
    result_list = [x for x in result_list if x != [None, None]]
    # create plots
    for i in range(len(result_list[0][0].fls)):
        plt.figure(i,figsize=(20,10))
        #add subplot in the corner
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
        ax1.set_title(gdir.rgi_id+' flowline '+str(i))
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        x = np.arange(y_2000.fls[i].nx) * y_2000.fls[i].dx * y_2000.fls[i].map_dx
        for j in range(len(result_list)):
            if result_list[j][0] != None:
                ax1.plot(x, result_list[j][0].fls[i].surface_h, alpha=0.8, )
                ax2.plot(x, result_list[j][1].fls[i].surface_h, alpha=0.8, )

        ax1.plot(x,y_1880.fls[i].surface_h,'k:')
        ax2.plot(x, y_2000.fls[i].surface_h, 'k')

        ax1.plot(x, y_1880.fls[i].bed_h, 'k')
        ax2.plot(x, y_2000.fls[i].bed_h, 'k')

        plot_dir = os.path.join(cfg.PATHS['working_dir'], 'plots','Ben_idea_parallel')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(os.path.join(plot_dir, gdir.rgi_id +'_fls'+str(i)+ '.png'))
    pickle.dump(result_list,open(os.path.join(plot_dir,gdir.rgi_id+'.pkl'),'wb'))



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
    cfg.PARAMS['use_intersects']=False
    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

    rgi = get_demo_file('rgi_oetztal.shp')
    gdirs = workflow.init_glacier_regions(salem.read_shapefile(rgi))
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    prepare_for_initializing(gdirs)
    '''
    pool = mp.Pool()
    pool.map(find_initial_state,gdirs)
    '''
    for gdir in gdirs:
        find_initial_state(gdir)


    print(time.time()-start_time)
