# Built ins
import os
from copy import deepcopy
from functools import partial

# External libs
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import pickle
import copy

# locals
import salem
from oggm import cfg, workflow, tasks, utils
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import PastMassBalance, RandomMassBalance, LinearMassBalance
from oggm.core.flowline import FluxBasedModel, FileModel
FlowlineModel = partial(FluxBasedModel, inplace=False)

def prepare_for_initializing(gdirs):
    '''
    oggm workflow for preparing initializing
    :param gdirs: list of oggm.GlacierDirectories
    :return None, but creates required files
    '''
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

    workflow.execute_entity_task(tasks.volume_inversion, gdirs)
    workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
    workflow.execute_entity_task(tasks.init_present_time_glacier,gdirs)

def _run_parallel_experiment(gdir):

    # read flowlines from pre-processing
    fls = gdir.read_pickle('model_flowlines')
    try:
        # construct searched glacier
        # TODO: y0 in random mass balance?
        random_climate1 = RandomMassBalance(gdir, y0=1850, bias=0, seed=[1])
        random_climate1.temp_bias = -0.75
        commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                      glen_a=cfg.A, y0=1850)
        commit_model.run_until_equilibrium()
        y_t0 = deepcopy(commit_model)

    # try different seed of mass balance, if equilibrium could not be found
    except:

        # construct searched glacier
        random_climate1 = RandomMassBalance(gdir, y0=1850, bias=0,seed=[1])
        commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                      glen_a=cfg.A, y0=1850)
        commit_model.run_until_equilibrium()
        y_t0 = deepcopy(commit_model)

    # construct observed glacier
    past_climate = PastMassBalance(gdir)
    commit_model2 = FluxBasedModel(commit_model.fls, mb_model=past_climate,
                                   glen_a=cfg.A, y0=1850)
    commit_model2.run_until(2000)
    y_t = deepcopy(commit_model2)

    # save output in gdir_dir
    experiment = {'y_t0': y_t0, 'y_t': y_t,
                  'climate': random_climate1}
    gdir.write_pickle(experiment, 'synthetic_experiment')

def synthetic_experiments(gdirs):
    '''
    creates searched and observed glacier to test the method, need only to
    be run once

    :param gdirs: list of oggm.GlacierDirectories
    :return:
    '''
    reset = True
    if os.path.isfile(gdirs[0].get_filepath('synthetic_experiment')):
        reset = utils.query_yes_no(
            'Running the function synthetic_experiments'
            ' will reset the previous results. Are you '
            ' sure you like to continue?')
    if not reset:
        return

    pool = mp.Pool()
    pool.map(_run_parallel_experiment, gdirs)
    pool.close()
    pool.join()


def find_possible_glaciers(gdir,experiment,y0):
    seed=1
    temp_bias = 0
    yr=2000

    fls = gdir.read_pickle('model_flowlines')
    #tasks.run_random_climate(gdir,nyears=500,y0=1850,bias=0,seed=seed,init_model_fls=copy.deepcopy(fls),output_filesuffix='_random_'+str(seed)+'_'+str(temp_bias))
    rp =  gdir.get_filepath('model_run', filesuffix='_random_'+str(seed)+'_'+str(temp_bias))
    fmod = FileModel(rp)
    fig, ax4 = plt.subplots()
    fmod.volume_m3_ts().mean().plot(ax=ax4)
    sorted = fmod.volume_m3_ts().sort_values()
    times=[]
    flowlines = []
    # TODO: erst ab hundert Jahre/ erstes großes lokales max. --> smooth und max bestimmen

    for q in np.linspace(0,1,9):
        times = np.append(times,[sorted[sorted >= sorted.quantile(q)].idxmin()])
    fig,ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    fig, ax3 = plt.subplots()
    for t in times:
        fmod.run_until(t)
        flowlines = np.append(flowlines,deepcopy(fmod.fls))
        ax1.plot(fmod.fls[-1].surface_h)
        ax4.plot(t,fmod.volume_m3,'o',markersize=10)

    for i,candidate_fls in enumerate(flowlines):
        #tasks.run_from_climate_data(gdir, ys=1850,ye=2000,
        #                         init_model_fls=copy.deepcopy(candidate_fls),
        #                         output_filesuffix='_past_' + str(
        #                             seed) + '_' + str(temp_bias)+'_'+str(i))
        pp = gdir.get_filepath('model_run',
                               filesuffix='_past_' + str(seed) + '_' +
                                          str(temp_bias)+'_'+str(i))
        fmod = FileModel(pp)
        fmod.run_until(fmod.last_yr)
        ax2.plot(fmod.fls[-1].surface_h)
        fmod.volume_m3_ts().plot(ax=ax3)
    ax2.plot(experiment['y_t'].fls[-1].surface_h,'k')

    ax1.plot(fls[-1].bed_h, 'k')
    ax2.plot(fls[-1].bed_h,'k')
    plt.show()

    return (seed, temp_bias, times)


if __name__ == '__main__':
    cfg.initialize()
    ON_CLUSTER = False
    # Local paths
    if ON_CLUSTER:
        cfg.PATHS['working_dir'] = os.environ.get("S_WORKDIR")
    else:
        #WORKING_DIR = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/opt_flowlines'
        WORKING_DIR = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/past_state_information'
        utils.mkdir(WORKING_DIR, reset=False)
        cfg.PATHS['working_dir'] = WORKING_DIR

    cfg.PATHS['plot_dir'] = os.path.join(cfg.PATHS['working_dir'], 'plots')

    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # How many grid points around the glacier?
    cfg.PARAMS['border'] = 80

    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['optimize_inversion_params'] = False
    cfg.PARAMS['use_intersects'] = False

    # add to BASENAMES
    _doc = 'contains observed and searched glacier from synthetic experiment to find intial state'
    cfg.BASENAMES['synthetic_experiment'] = ('synthetic_experiment.pkl', _doc)
    _doc = 'output of reconstruction'
    cfg.BASENAMES['reconstruction_output'] = (
    'reconstruction_output.pkl', _doc)

    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

    rgi = get_demo_file('rgi_oetztal.shp')
    rgidf = salem.read_shapefile(rgi)
    gdirs = workflow.init_glacier_regions(rgidf[rgidf.RGIId != 'RGI50-11.00779'])
    #gdirs = workflow.init_glacier_regions(rgidf)

    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
    #prepare_for_initializing(gdirs)

    for gdir in gdirs[:1]:
        fig,ax1 = plt.subplots()
        fig, ax2 = plt.subplots()
        #synthetic_experiments([gdir])
        experiment = gdir.read_pickle('synthetic_experiment')
        fls = gdir.read_pickle('model_flowlines')

        seed,temp,times = find_possible_glaciers(gdir,experiment,1850)

        '''

        # ---------------------- EXPERIMENT MODEL ----------------------
        #model = tasks.run_from_climate_data(gdir,ys=1850,ye=2000,init_model_fls=deepcopy(experiment['y_t0'].fls),output_filesuffix='_experiment_past')

        # Model from file
        fp = gdir.get_filepath('model_run', filesuffix='_experiment_past')
        fmod = FileModel(fp)

        # "READ" last year state
        fmod.run_until(fmod.last_yr)
        ax2.plot(fmod.fls[-1].surface_h, label='experiment')

        # plot volume over hole time intervall
        fmod.volume_m3_ts().plot(ax=ax1, label='experiment')

        # ---------------------- DEFAULT MODEL ----------------------
        #model = tasks.run_from_climate_data(gdir, ys=1850, ye=2000, init_model_fls=fls,output_filesuffix='_default_past')
        fp = gdir.get_filepath('model_run', filesuffix='_default_past')
        fmod = FileModel(fp)

        # "READ" last year state
        fmod.run_until(fmod.last_yr)
        ax2.plot(fmod.fls[-1].surface_h, label='default glacier')

        # plot volume over hole time intervall
        fmod.volume_m3_ts().plot(ax=ax1, label='default glacier')

        # ---------------------- ZERO - GLACIER ----------------------
        #model = tasks.run_from_climate_data(gdir, ys=1850, ye=2000, zero_initial_glacier=True,output_filesuffix='_zero_past')
        fp = gdir.get_filepath('model_run', filesuffix='_zero_past')
        fmod = FileModel(fp)

        # "READ" last year state
        fmod.run_until(fmod.last_yr)
        ax2.plot(fmod.fls[-1].surface_h, label='zero glacier')

        # plot volume over hole time intervall
        fmod.volume_m3_ts().plot(ax=ax1, label='zero glacier')

        ax2.plot(fls[-1].bed_h,'k')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.show()
        #model = FluxBasedModel(deepcopy(experiment['y_t0'].fls),PastMassBalance(gdir,bias=0),y0=1850)
        #a,b = model.run_until_and_store(2000)

        #b.volume_m3.plot(ax=ax1,color='k', label='experiment')
        #ax2.plot(model.fls[-1].surface_h, 'k',label='experiment')


        #b.sortby('volume_m3').volume_m3
        '''
        #plt.show()
