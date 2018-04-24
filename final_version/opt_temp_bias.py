
# Built ins
import os
import copy
from functools import partial

# External libs
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import pickle

# locals
import salem
from oggm import cfg, workflow, tasks, utils
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
FlowlineModel = partial(FluxBasedModel, inplace=False)

from final_version.plots import plot_experiment,plot_surface, make_result_panda


def objfunc(param, gdir, y_2000, random_climate2):
    '''
    calculate difference between predicted and observerd glacier
    :param param: temp bias to be optimized
    :param gdir: oggm.GlacierDirectory
    :param y_2000: oggm.Flowline of observed state
    :param random_climate2: oggm.massbalance.RandomMassBalance object
    :return: objective value
    '''
    try:
        random_model, past_model = run_model(param, gdir, y_2000, random_climate2)
        f = np.sum(abs(past_model.fls[-1].surface_h -y_2000[-1].surface_h)**2) + \
            np.sum(abs(past_model.fls[-1].widths - y_2000[-1].widths) ** 2)
            #abs(real_model.length_m - y_2000.length_m)**2
            #abs(real_model.area_m2 - y_1900.area_m2)**2 + \
            #abs(real_model.volume_m3-y_1900.volume_m3)**2
    except:
        f=1e10
    print(param, f)
    return f

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

    workflow.execute_entity_task(tasks.optimize_inversion_params,gdirs)
    workflow.execute_entity_task(tasks.volume_inversion, gdirs)
    workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
    workflow.execute_entity_task(tasks.init_present_time_glacier,gdirs)


def run_model(param,gdir,y_t,random_climate2):
    '''
    :param param: temp bias, is changed by optimization
    :param gdir:  oggm.GlacierDirectory
    :param y_2000: oggm.Flowline for the observed state (2000)
    :param random_climate2: oggm.massbalance.RandomMassBalance
    :return: 2 oggm.flowline.FluxBasedModels
             (glacier candidate and predicted glacier model)
    '''

    # run estimated glacier with random climate 2 until equilibrium
    # (glacier candidate)

    # estimated flowline = observed flowline
    estimated_fls = copy.deepcopy(y_t)
    climate = copy.deepcopy(random_climate2)
    # change temp_bias
    climate.temp_bias = param
    random_model = FluxBasedModel(estimated_fls, mb_model=climate, y0=1865)
    random_model.run_until_equilibrium()

    # run glacier candidate with past climate until 2000
    candidate_fls= copy.deepcopy(y_t)
    for i in range(len(y_t)):
        candidate_fls[i].surface_h = random_model.fls[i].surface_h

    past_climate = PastMassBalance(gdir)
    past_model = FluxBasedModel(candidate_fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)
    past_model.run_until(2000)

    return [random_model,past_model]


def run_optimization(gdirs):

    for gdir in gdirs:
        experiments = gdir.read_pickle('synthetic_experiment')
        y_t = experiments['y_t'].fls
        pool = mp.Pool()
        result_list = pool.map(partial(run_parallel, gdir=gdir, y_t=y_t),
                               range(4))
        gdir.write_pickle(result_list,'reconstruction_output')


def run_parallel(i,gdir,y_t):
    try:
        random_climate2 = RandomMassBalance(gdir, y0=1865, halfsize=14)
        experiment = gdir.read_pickle('synthetic_experiment')
        # ensure that not the same climate as in the experiments is used
        if experiment['climate']==random_climate2:
            random_climate2 = RandomMassBalance(gdir, y0=1865, halfsize=14)
        res = minimize(objfunc, [0],
                       args=(gdir, y_t,random_climate2),
                       method='COBYLA',
                       tol=1e-04, options={'maxiter': 100, 'rhobeg': 1})

        result_model_t0, result_model_t = run_model(res.x, gdir,
                                                         y_t,random_climate2)
        return result_model_t0,result_model_t
    except:
        return None, None


def _run_parallel_experiment(gdir):

    # read flowlines from pre-processing
    fls = gdir.read_pickle('model_flowlines')
    try:
        # construct searched glacier
        random_climate1 = RandomMassBalance(gdir, y0=1865, halfsize=14)
        commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                      glen_a=cfg.A, y0=1850)
        commit_model.run_until_equilibrium()
        y_t0 = copy.deepcopy(commit_model)

    # try different seed of mass balance, if equilibrium could not be found
    except:
        # construct searched glacier
        random_climate1 = RandomMassBalance(gdir, y0=1865, halfsize=14)
        commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                      glen_a=cfg.A, y0=1850)
        commit_model.run_until_equilibrium()
        y_t0 = copy.deepcopy(commit_model)

    # construct observed glacier
    past_climate = PastMassBalance(gdir)
    commit_model2 = FluxBasedModel(commit_model.fls, mb_model=past_climate,
                                   glen_a=cfg.A, y0=1865)
    commit_model2.run_until(2000)
    y_t = copy.deepcopy(commit_model2)

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

        reset = utils.query_yes_no('Running the function synthetic_experiments'
                                   ' will reset the previous results. Are you '
                                   ' sure you like to continue?')
    if not reset:
        return

    pool = mp.Pool()
    pool.map(_run_parallel_experiment,gdirs)


if __name__ == '__main__':
    cfg.initialize()
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
    #cfg.PATHS['working_dir'] = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state'
    #cfg.PATHS['plot_dir'] = os.path.join(cfg.PATHS['working_dir'],'plots')
    cfg.PATHS['working_dir'] = os.environ.get("S_WORKDIR")
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['prcp_scaling_factor']
    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['optimize_inversion_params'] = True
    cfg.PARAMS['use_intersects'] = False
    # add to BASENAMES
    _doc = 'contains observed and searched glacier from synthetic experiment to find intial state'
    cfg.BASENAMES['synthetic_experiment'] = ('synthetic_experiment.pkl',_doc)
    _doc = 'output of reconstruction'
    cfg.BASENAMES['reconstruction_output'] =('reconstruction_output.pkl',_doc)

    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

    rgi = get_demo_file('rgi_oetztal.shp')
    gdirs = workflow.init_glacier_regions(salem.read_shapefile(rgi))
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
    prepare_for_initializing(gdirs)
    synthetic_experiments(gdirs)
    run_optimization(gdirs)

    #for gdir in gdirs:
    #    plot_experiment(gdir,cfg.PATHS['plot_dir'])
    #    plot_surface(gdir,cfg.PATHS['plot_dir'],-1)
