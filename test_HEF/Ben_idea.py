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


def run_model(param,gdir,fls):

    fls1 = copy.deepcopy(fls)
    climate = random_climate2
    climate.temp_bias = param
    model = FluxBasedModel(fls1, mb_model=climate, y0=1890)
    model.run_until_equilibrium()

    # fls2= copy.deepcopy(fls)
    fls2 = model.fls

    real_model = FluxBasedModel(fls2, mb_model=past_climate,
                                glen_a=cfg.A, y0=1850)

    real_model.run_until(2000)

    return [model,real_model]

def objfunc(param,gdir,fls):

    try:
        model, real_model = run_model(param,gdir,fls)

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

    for i in range(30):
        random_climate2 = RandomMassBalance(gdir, y0=1875, halfsize=14)
        res = minimize(objfunc, [0], args=(gdir, y_2000.fls),
                       method='COBYLA',
                       tol=1e-04, options={'maxiter': 100, 'rhobeg': 1})
        #try:
        result_model_1880, result_model_2000 = run_model(res.x, gdir,
                                                         y_2000.fls)
        results = results.append({'1880':result_model_1880,'2000':result_model_2000,'length_1880':result_model_1880.length_m,'length_2000':result_model_2000.length_m}, ignore_index=True)
        #except:
        #   pass

    # create plots
    for i in range(len(fls)):
        plt.figure(i,figsize=(20,10))
        #add subplot in the corner
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
        ax1.set_title(gdir.rgi_id+' flowline '+str(i))
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        x = np.arange(y_2000.fls[i].nx) * y_2000.fls[i].dx * y_2000.fls[i].map_dx
        for j in range(len(results)):
            ax1.plot(x, results.get_value(j, '1880').fls[i].surface_h, alpha=0.8, )
            ax2.plot(x, results.get_value(j, '2000').fls[i].surface_h, alpha=0.8, )

        ax1.plot(x,y_1880.fls[i].surface_h,'k:')
        ax2.plot(x, y_2000.fls[i].surface_h, 'k')

        ax1.plot(x, y_1880.fls[i].bed_h, 'k')
        ax2.plot(x, y_2000.fls[i].bed_h, 'k')

        plot_dir = os.path.join(cfg.PATHS['working_dir'], 'plots')
        plt.savefig(os.path.join(plot_dir, gdir.rgi_id +'_fls'+str(i)+ '.png'))
    #plt.show()

    results.to_csv(os.path.join(plot_dir,str(gdir.rgi_id)+'.csv'))
    '''
    # prepare plots
    #plt.figure(figsize=(20, 10))
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    ax1.set_title(gdir.rgi_id)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    fig2=plt.figure()
    ax4 = fig2.add_subplot(1, 1, 1)
    fig3=plt.figure()
    ax5 = fig3.add_subplot(1, 1, 1)



    x = np.arange(y_2000.fls[-1].nx) * y_2000.fls[-1].dx * y_2000.fls[
        -1].map_dx
    x1=np.arange(y_2000.fls[-2].nx) * y_2000.fls[-2].dx * y_2000.fls[
        -2].map_dx
    x2 = np.arange(y_2000.fls[-3].nx) * y_2000.fls[-3].dx * y_2000.fls[
        -3].map_dx


    for i in range(1):
        random_climate2 = RandomMassBalance(gdir, y0=1875, halfsize=14)
        res = minimize(objfunc, [0], args=(gdir, y_2000.fls),
                       method='COBYLA',
                       tol=1e-04, options={'maxiter': 100, 'rhobeg': 1})


        try:
            result_model_1850, result_model_1900 = run_model(res.x, gdir,
                                                             y_2000.fls)


            dif_s = result_model_1900.fls[-1].surface_h - y_2000.fls[-1].surface_h
            dif_w = result_model_1900.fls[-1].widths - y_2000.fls[-1].widths
            # if np.max(dif_s)<40 and np.max(dif_w)<15:

            ax1.plot(x, result_model_1850.fls[-1].surface_h, alpha=0.8,)
            ax2.plot(x, result_model_1900.fls[-1].surface_h, alpha=0.8)
            ax4.plot(result_model_1900.fls[-2].widths, alpha=0.8)
            ax5.plot(result_model_1900.fls[-3].widths, alpha=0.8)

        except:
            pass


    ax1.plot(x, y_1880.fls[-1].surface_h,
             'k:')  # , label='surface elevation (not known)')
    ax1.plot(x, y_1880.fls[-1].bed_h, 'k')  # , label='bed topography')
    ax2.plot(x, y_2000.fls[-1].surface_h, 'k',
             label='surface elevation (observed)')
    ax4.plot( y_2000.fls[-2].widths, 'k:',
             label='surface elevation (observed)')

    ax5.plot( y_2000.fls[-3].widths, 'k:',
             label='surface elevation (observed)')

    ax2.plot(x, y_2000.fls[-1].bed_h, 'k', label='bed')
    # ax3.plot(x,np.zeros(len(x)),'k:')
    ax1.annotate('t = 1850', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=13)
    ax2.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',
                 fontsize=9)
    ax1.set_xlabel('Distance along the Flowline (m)')
    ax1.set_ylabel('Altitude (m)')

    ax2.set_xlabel('Distance along the Flowline (m)')
    ax2.set_ylabel('Altitude (m)')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='best')
    plot_dir = os.path.join(cfg.PATHS['working_dir'], 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, gdir.rgi_id + '.png'))
    plt.show()
    '''


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

    pool = mp.Pool()
    pool.map(find_initial_state,gdirs)
    '''
    for gdir in gdirs:
        if gdir.rgi_id == "RGI50-11.00897":
            find_initial_state(gdir)
    '''

    print(time.time()-start_time)
