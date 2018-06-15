
# Built ins
import os
import sys
from copy import deepcopy
from functools import partial

# Module logger
import logging
log = logging.getLogger(__name__)

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
from oggm.core.massbalance import PastMassBalance, RandomMassBalance, LinearMassBalance
from oggm.core.flowline import FluxBasedModel
FlowlineModel = partial(FluxBasedModel, inplace=False)

#from final_version.multistep.plots import *


def find_best_objective(gdir,y_te,t0,te):
    results = gdir.read_pickle('reconstruction_output')
    df = pd.DataFrame(columns=[str(t0),str(te)+'_obs'])
    for res in results:
        if res[0] != None:
            df = df.append(pd.DataFrame([res],columns=[str(t0),str(te)+'_obs']),
                           ignore_index=True)
    df['objective_'+str(te)] = df[str(te)+'_obs'].apply((lambda x: objective_value(y_te,x.fls)))
    df['objective_'+str(te)].idxmin
    return df, df['objective_'+str(te)].idxmin

def objective_value(fls1,fls2):
    return(np.sum(abs(fls1[-1].surface_h-fls2[-1].surface_h)**2)+ \
          np.sum(abs(fls1[-1].widths-fls2[-1].widths)**2))

def objfunc(param, gdir, y_t, random_climate2,t0,te):
    '''
    calculate difference between predicted and observerd glacier
    :param param: temp bias to be optimized
    :param gdir: oggm.GlacierDirectory
    :param y_2000: oggm.Flowline of observed state
    :param random_climate2: oggm.massbalance.RandomMassBalance object
    :return: objective value
    '''
    try:

        random_model, past_model = run_model(param, gdir, y_t, random_climate2,
                                             t0,te)
        f = objective_value(past_model.fls,y_t)
        #f = np.sum(abs(past_model.fls[-1].surface_h -y_t[-1].surface_h)**2) + \
        #    np.sum(abs(past_model.fls[-1].widths - y_t[-1].widths) ** 2)
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
        tasks.glacier_masks,
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


def run_model(param,gdir,y_t,random_climate2,t0,te):
    '''
    :param param: temp bias, is changed by optimization
    :param gdir:  oggm.GlacierDirectory
    :param y_t: oggm.Flowline for the observed state (2000)
    :param random_climate2: oggm.massbalance.RandomMassBalance
    :return: 2 oggm.flowline.FluxBasedModels
             (glacier candidate and predicted glacier model)
    '''

    # run estimated glacier with random climate 2 until equilibrium
    # (glacier candidate)


    # estimated flowline = observed flowline
    estimated_fls = deepcopy(y_t)
    climate = deepcopy(random_climate2)
    # change temp_bias
    climate.temp_bias = param
    random_model = FluxBasedModel(estimated_fls, mb_model=climate, y0=t0)
    random_model.run_until_equilibrium()
    # run glacier candidate with past climate until 2000
    candidate_fls= deepcopy(y_t)
    for i in range(len(y_t)):
        candidate_fls[i].surface_h = random_model.fls[i].surface_h


    past_climate = PastMassBalance(gdir)
    past_model = FluxBasedModel(candidate_fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    past_model.run_until(te)

    return [random_model,past_model]


def con(param):
    return -param


def run_optimization(gdir,t0,te,y_te):
    ''' run optimization
    '''


    pool = mp.Pool()
    result_list = pool.map(partial(run_parallel, gdir=gdir, y_t=y_te,t0=t0,
                                   te=te), range(56))
    pool.close()
    pool.join()
    gdir.write_pickle(result_list,'reconstruction_output')

def run_parallel(i,gdir,y_t,t0,te):
    try:
        random_climate2 = RandomMassBalance(gdir, y0=t0, halfsize=14)
        res = minimize(objfunc, [0],
                       args=(gdir, y_t,random_climate2,t0,te),
                       method='COBYLA',
                       #constraints={'type':'ineq','fun':con},
                       tol=1e-04, options={'maxiter': 100, 'rhobeg': 1})

        result_model_t0, result_model_t = run_model(res.x, gdir, y_t,
                                                    random_climate2, t0, te)
        return result_model_t0,result_model_t
    except:
        return None, None


def _run_parallel_experiment(gdir,t0,te):
    '''

    :param gdir:
    :param t0:
    :param te:
    :return:
    '''

    # read flowlines from pre-processing
    fls = gdir.read_pickle('model_flowlines')
    try:
        # construct searched glacier
        random_climate1 = RandomMassBalance(gdir, y0=t0, halfsize=14)

        #set temp bias negative to force a glacier retreat later
        random_climate1.temp_bias= -0.75
        commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                      glen_a=cfg.A, y0=t0)
        commit_model.run_until_equilibrium()
        y_t0 = deepcopy(commit_model)

    # try different seed of mass balance, if equilibrium could not be found
    except:

        # construct searched glacier
        random_climate1 = RandomMassBalance(gdir, y0=t0, halfsize=14)
        commit_model = FluxBasedModel(fls, mb_model=random_climate1,
                                      glen_a=cfg.A, y0=t0)
        commit_model.run_until_equilibrium()
        y_t0 = deepcopy(commit_model)

    # construct observed glacier
    past_climate = PastMassBalance(gdir)
    commit_model2 = FluxBasedModel(commit_model.fls, mb_model=past_climate,
                                   glen_a=cfg.A, y0=t0)
    commit_model2.run_until(te)
    y_t = deepcopy(commit_model2)

    # save output in gdir_dir
    experiment = {'y_t0': y_t0, 'y_t': y_t,
                  'climate': random_climate1}
    gdir.write_pickle(experiment, 'synthetic_experiment')

def synthetic_experiments(gdirs,t0,te):
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
    pool.map(partial(_run_parallel_experiment,t0=t0,te=te),gdirs)
    pool.close()
    pool.join()


if __name__ == '__main__':

    #initialize OGGM and set up the default run parameters
    cfg.initialize()
    ON_CLUSTER = False
    #Local paths
    if ON_CLUSTER:
        cfg.PATHS['working_dir'] = os.environ.get("S_WORKDIR")
    else:
        WORKING_DIR = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/retreat3'
        utils.mkdir(WORKING_DIR, reset=False)
        cfg.PATHS['working_dir'] = WORKING_DIR

    cfg.PATHS['plot_dir'] =os.path.join(cfg.PATHS['working_dir'],'plots')

    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')

    #Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # How many grid points around the glacier?
    cfg.PARAMS['border'] = 80

    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['optimize_inversion_params'] = False
    cfg.PARAMS['use_intersects'] = False

    # add to BASENAMES
    _doc = 'contains observed and searched glacier from synthetic experiment to find intial state'
    cfg.BASENAMES['synthetic_experiment'] = ('synthetic_experiment.pkl',_doc)
    _doc = 'output of reconstruction'
    cfg.BASENAMES['reconstruction_output'] =('reconstruction_output.pkl',_doc)

    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size

    # get rgi file
    rgi = get_demo_file('rgi_oetztal.shp')
    rgidf = salem.read_shapefile(rgi)

    # Initialize working directories
    gdir = workflow.init_glacier_regions(rgidf[rgidf.RGIId== 'RGI50-11.00897'])[0]
    #prepare_for_initializing([gdir])

    result = pd.Series()
    fls = gdir.read_pickle('model_flowlines')
    '''
    fls_obs = deepcopy(fls)
    i = 1
    for yr in np.arange(2000,1850,-50):
        t0 = yr-50
        te = yr
        run_optimization(gdir, t0, te, fls_obs)
        df, best = find_best_objective(gdir,fls,t0,te)
        result = result.append(df,ignore_index=True)
        pickle.dump(result, open(os.path.join(gdir.dir,'result_multistep'+str(i)),'wb'))
        fls_obs = df.loc[best,str(t0)].fls
        i=i+1

    '''
    result = pickle.load(open(os.path.join(gdir.dir,'result_multistep3'),'rb'))
    x = np.arange(fls[-1].nx) * fls[-1].dx * fls[-1].map_dx
    i = 0
    plt.figure()
    grid = plt.GridSpec(2, 4, wspace=0.4, hspace=0.3)

    for yr in np.arange(1850,2001, 50):
        plt.subplot(grid[0,i])
        if yr !=1850:
            plt.plot(x,result[str(yr)+'_obs'].dropna().loc[best_id].fls[-1].surface_h,
                     'b')
        if yr != 2000:
            for model in result[str(yr)].dropna():
                plt.plot(x,model.fls[-1].surface_h,'grey',alpha=0.5)
            best_id = result['objective_' + str(yr+50)].idxmin()
            plt.plot(x,result[str(yr)].dropna().loc[best_id].fls[-1].surface_h, 'r')
        if yr == 2000:
            plt.plot(x,fls[-1].surface_h,'k')
        plt.plot(x,model.fls[-1].bed_h,'k')
        i = i+1


    plt.subplot(grid[1,:])
    for yr in np.arange(1850, 2000, 50):
        plt.axvline(x=yr, color='k')
        best_id = result['objective_' + str(yr + 50)].idxmin()
        list = result['objective_' + str(yr + 50)].dropna().index
        if not best_id in list:
            list = list.append(pd.Index([best_id]))
        for i in list:

            fls = deepcopy(result[str(yr)].dropna().loc[i].fls)
            past_climate = PastMassBalance(gdir)
            model = FluxBasedModel(fls, mb_model=past_climate,
                                 glen_a=cfg.A, y0=yr)
            if i == best_id:
                plt.plot(model.yr,model.length_m,'ro')
                a, b = model.run_until_and_store(yr+50)
                b.length_m.plot(color='red')
                plt.plot(model.yr, model.length_m, 'bo')
            else:
                a, b = model.run_until_and_store(yr+50)
                b.length_m.plot( color='grey', alpha=0.3)
        if yr==1850:
            a,b = model.run_until_and_store(2000)
            b.length_m.plot(linestyle=':', color='red')


    plt.xlim((1850, 2000))
    plt.show()
    '''
    plt.figure(1)


    plt.subplot(331)
    for model in result['1900'].dropna():
        plt.plot(x,model.fls[-1].surface_h, 'grey')
    best = result['objective_1950'].dropna().idxmin
    plt.plot(x,result['1900'].dropna().loc[best].fls[-1].surface_h, 'r')
    plt.plot(x,result['1950'].dropna().loc[0].fls[-1].bed_h, 'k')
    plt.title('1900')


    plt.subplot(332)
    for model in result['1950'].dropna():
        plt.plot(x,model.fls[-1].surface_h,'grey')
    best = result['objective_2000'].dropna().idxmin
    plt.plot(x,result['1950'].dropna().loc[best].fls[-1].surface_h, 'r')
    best2 = result['objective_1950'].dropna().idxmin
    plt.plot(x, result['1950_obs'].dropna().loc[best2].fls[-1].surface_h, 'k')
    plt.plot(x,result['1950'].dropna().loc[0].fls[-1].bed_h,'k')
    plt.title(1950)

    plt.subplot(333)
    for model in result['2000_obs'].dropna():
        plt.plot(x,model.fls[-1].surface_h, 'grey')
    best = result['objective_2000'].dropna().idxmin
    plt.plot(x,result['2000_obs'].dropna().loc[best].fls[-1].surface_h, 'r')
    plt.plot(x,fls[-1].surface_h,'k')
    plt.plot(x,result['1950'].dropna().loc[0].fls[-1].bed_h, 'k')
    plt.title(2000)



    plt.subplot(312)
    plt.axvline(x=1950,color='k')
    past_climate = PastMassBalance(gdir)



    obs = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=2000)
    plt.plot(2000,obs.length_m,'ko')
    best_id = result['objective_2000'].idxmin()
    for index, model in result.head(10).iterrows():
        if type(model['1950']) != float:
            past_model = FluxBasedModel(model['1950'].fls, mb_model=past_climate,
                                        glen_a=cfg.A, y0=1950)
            l_t0 = past_model.length_m
            a, b = past_model.run_until_and_store(2000)
            if index != best_id:
                b.length_m.plot(color='grey',alpha=0.5)
                plt.plot(1950,l_t0,marker='o',color='grey')
            else:
                b.length_m.plot(color='red')
                plt.plot(1950, l_t0, 'ro')


    best_id = result['objective_1950'].idxmin()
    for index, model in result.head(60).iterrows():
        if type(model['1900']) != float:
            past_model = FluxBasedModel(model['1900'].fls, mb_model=past_climate,
                                        glen_a=cfg.A, y0=1900)
            l_t0 = past_model.length_m
            a, b = past_model.run_until_and_store(1950)
            if index != best_id:
                b.length_m.plot(color='grey', alpha=0.5)
                plt.plot(1900,l_t0,marker='o',color='grey')
            else:
                b.length_m.plot(color='red')
                plt.plot(1900, l_t0, 'ro')


    plt.xlim((1850,2001))


    plt.subplot(337)
    for model in result['1900'].dropna():
        plt.plot(x, model.fls[-1].widths, 'grey')
    best = result['objective_1950'].dropna().idxmin
    plt.plot(x, result['1900'].dropna().loc[best].fls[-1].widths, 'r')
    plt.title('1900')

    plt.subplot(338)
    for model in result['1950'].dropna():
        plt.plot(x, model.fls[-1].widths, 'grey')
    best = result['objective_2000'].dropna().idxmin
    plt.plot(x, result['1950'].dropna().loc[best].fls[-1].widths, 'r')
    best2 = result['objective_1950'].dropna().idxmin
    plt.plot(x, result['1950_obs'].dropna().loc[best2].fls[-1].widths, 'k')
    plt.title(1950)

    plt.subplot(339)
    for model in result['2000_obs'].dropna():
        plt.plot(x, model.fls[-1].widths, 'grey')
    best = result['objective_2000'].dropna().idxmin
    plt.plot(x, result['2000_obs'].dropna().loc[best].fls[-1].widths, 'r')
    plt.plot(x, fls[-1].widths, 'k')
    plt.title(2000)

    plt.show()
    '''


