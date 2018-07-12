# Built ins
import os
from copy import deepcopy
from functools import partial

# External libs
import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import pickle
import copy
from multiprocessing import Pool

# locals
import salem
from oggm import cfg, workflow, tasks, utils, graphics
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel, FileModel
FlowlineModel = partial(FluxBasedModel, inplace=False)
#from final_version.past_state_information.plots import plot_candidates, plot_surface, plot_lenght, plot_largest_glacier,plot_surface_col,plot_volume_dif_time

def objective_value(model1,model2):
    fls1 = model1.fls
    fls2 = model2.fls
    return(np.sum(abs(fls1[-1].surface_h-fls2[-1].surface_h)**2)+ \
          np.sum(abs(fls1[-1].widths-fls2[-1].widths)**2))

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


def run_to_present(tupel,gdir,ys,ye):
    suffix = tupel[0]
    path = gdir.get_filepath('model_run',filesuffix=suffix)
    # does file already exists?
    if not os.path.exists(path):
        try:
            model = tasks.run_from_climate_data(gdir, ys=ys, ye=ye,
                                                output_filesuffix=suffix,
                                                init_model_fls=copy.deepcopy(
                                                    tupel[1].fls))
            return suffix
        # oggm failed --> probaly "glacier exeeds boundaries"
        except:
            return None

    else:
        # does file contain a model?
        try:
            fmod = FileModel(path)
            return suffix
        except:
            return None

def read_file_model(suffix,gdir):
    rp = gdir.get_filepath('model_run',filesuffix=suffix)
    fmod = FileModel(rp)
    return copy.deepcopy(fmod)

def run_file_model(suffix,ye):
    rp = gdir.get_filepath('model_run', filesuffix=suffix)
    fmod = FileModel(rp)
    fmod.run_until(ye)
    return copy.deepcopy(fmod)


def find_candidates(gdir, experiment, df, ys,ye,n):
    indices = []
    for q in np.linspace(0,1,n):
        # indices of all to test
        index = df[df['ts_section'] >= (df['ts_section'].quantile(q))]['ts_section'].idxmin()
        indices = np.append(indices,int(index))
    candidates = df.ix[indices]
    candidates = candidates.sort_values(['suffix','time'])
    candidates['fls_t0']=None
    for suffix in candidates['suffix'].unique():
        rp = gdir.get_filepath('model_run', filesuffix=suffix)
        fmod = FileModel(rp)
        for i,t in candidates[candidates['suffix']==suffix]['time'].iteritems():
            fmod.run_until(t)
            candidates.at[i,'random_model_t0'] = copy.deepcopy(fmod)

    candidates = candidates.drop_duplicates()
    fls_list =[]
    for i in candidates.index:
        s = candidates.loc[int(i),'suffix'].split('_random')[-1]
        suffix = str(ys)+'_past'+s+'_'+str(int(candidates.loc[int(i),'time']))
        fls = candidates.loc[int(i),'random_model_t0']
        fls_list.append([suffix,fls])


    # run candidates until present
    pool = Pool()
    path_list = pool.map(partial(run_to_present, gdir=gdir, ys=ys,
                                     ye=2000), fls_list)


    pool.close()
    pool.join()

    candidates = candidates.assign(past_suffix=path_list)
    candidates['model_t0'] = candidates['past_suffix'].apply(read_file_model,
                                                             args=([gdir]))
    candidates['model_t'] = candidates['past_suffix'].apply(run_file_model,
                                                         args=([ye]))
    candidates['objective'] = candidates['model_t'].apply(objective_value,
                                                          args=([experiment['y_t']]))

    return candidates

def run_random_task(tupel,gdir,y0):
    seed = tupel[0]
    temp_bias = tupel[1]
    fls = gdir.read_pickle('model_flowlines')
    suffix = str(y0)+'_random_'+str(seed)+'_'+ str(temp_bias)

    # test if file already exist:
    path = gdir.get_filepath('model_run', filesuffix=suffix)

    # does file already exists?
    if not os.path.exists(path):
        try:
            tasks.run_random_climate(gdir, nyears=400, y0=y0, bias=0,
                                     seed=seed, temperature_bias=temp_bias,
                                     init_model_fls=copy.deepcopy(fls),
                                     output_filesuffix=suffix)
            return path
        # oggm failed --> probaly "glacier exeeds boundaries"
        except:
            return None

    else:
        # does file contain a model?
        try:
            fmod = FileModel(path)
            return path
        except:
            return None

def run_random_parallel(gdir,y0,list):
    pool = Pool()
    paths = pool.map(partial(run_random_task, gdir=gdir, y0=y0), list)
    pool.close()
    pool.join()

    random_run_list = pd.DataFrame()
    for rp in paths:
        if rp != None:
            temp_bias = rp.split('.nc')[0].split('_')[-1]
            seed = rp.split('.nc')[0].split('_')[-2]
            suffix = str(y0)+'_random_'+str(seed)+'_'+ str(temp_bias)
            v = pd.Series({'seed': seed, 'temp_bias': float(temp_bias),'suffix':suffix})
            random_run_list = random_run_list.append(v, ignore_index=True)
    return random_run_list

def find_temp_bias_range(gdir,y0):
    fls = gdir.read_pickle('model_flowlines')
    t_eq = 0

    # try range (2,-2) first
    bias_list= [b.round(3) for b in np.arange(-2,2,0.05)]
    list = [(i**2, b) for i,b in enumerate(bias_list)]
    random_run_list = run_random_parallel(gdir,y0,list)


    # smaller temperature bias is still possible to test
    if random_run_list['temp_bias'].min()==-2:
        n = len(random_run_list)
        list = [((i+n+1)**2, b.round(3)) for i, b in enumerate(np.arange(-3,-2.05,0.05))]
        random_run_list = random_run_list.append(run_random_parallel(gdir, y0, list),ignore_index=True)

    # check for zero glacier
    max_bias = random_run_list['temp_bias'].idxmax()
    p = gdir.get_filepath('model_run', filesuffix=random_run_list.loc[max_bias,'suffix'])
    fmod = FileModel(p)

    if not fmod.volume_m3_ts().min()==0:
        n = len(random_run_list)
        list = [((i + n + 1) ** 2, b.round(3)) for i, b in enumerate(np.arange(2.05,3, 0.05))]
        random_run_list = random_run_list.append(run_random_parallel(gdir, y0, list), ignore_index=True)

    # find t_eq
    for suffix in random_run_list['suffix'].head(10).values:
        rp = gdir.get_filepath('model_run', filesuffix=suffix)
        fmod = FileModel(rp)
        t = _find_t_eq(fmod.volume_m3_ts())
        if t > t_eq:
            t_eq = t

    all = pd.DataFrame()
    for suffix in random_run_list['suffix']:
        rp = gdir.get_filepath('model_run',filesuffix=suffix)
        fmod = FileModel(rp)
        v = pd.DataFrame(fmod.volume_m3_ts()).reset_index()
        v = v[v['time']>=t_eq]
        v = v.assign(suffix=lambda x: suffix)
        all = all.append(v, ignore_index=True)
    return all


def _find_t_eq(df):
    # smooth to find maximum
    smooth_df = df.rolling(10).mean()
    # fill nan with 0, does not change the extrema, but avoid warning from np.greater
    smooth_df = smooth_df.fillna(0)
    # find extrema and take the first one
    extrema = argrelextrema(smooth_df.values, np.greater,order=20)[0][0]

    return extrema

def find_possible_glaciers(gdir,experiment,y0):
    seed=1
    temp_bias=0
    temp_bias_list1 = [1.0,0.75,0.5,0.25,0.0,-0.25,-0.5,-0.7]
    temp_bias_list2 = [-0.75,-0.8,-0.85,-0.9,-0.95]
    yr=2000
    # find good temp_bias_list
    random_df = find_temp_bias_range(gdir,y0)
    cand_df = find_candidates(gdir, experiment, df=random_df, ys=y0, ye=2000, n=200)

    return cand_df

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
    cfg.PARAMS['border'] = 150

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
    gdirs = workflow.init_glacier_regions(rgidf)
    #gdirs = workflow.init_glacier_regions(rgidf)

    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
    prepare_for_initializing(gdirs)

    for gdir in gdirs:
        #if gdir.rgi_id not in ['RGI50-11.00945','RGI50-11.00779']:
        if gdir.rgi_id == 'RGI50-11.00698':
            #fig,ax1 = plt.subplots()
            #fig, ax2 = plt.subplots()
            try:
                experiment = gdir.read_pickle('synthetic_experiment')
            except:
                synthetic_experiments([gdir])
                experiment = gdir.read_pickle('synthetic_experiment')
            fls = gdir.read_pickle('model_flowlines')
            yrs = np.arange(1850,1950,10)
            results={}
            for yr in yrs:

                candidates_df = find_possible_glaciers(gdir,experiment,yr)
                results[yr] = candidates_df
            pickle.dump(results,open(os.path.join(gdir.dir,'results.pkl'),'wb'))
                #plot_surface_col(gdir, candidates_df, experiment, yr)

            #plot_volume_dif_time(gdir, results, experiment)
            # get only glaciers, that are equal to observation
            #plot_candidates(gdir, candidates_df, experiment, ys)
            #plot_surface_col(gdir, val, experiment, key)
            # plot_lenght(gdir,candidates,experiment)
            # plot_surface(gdir,candidates,experiment)
            #plot_largest_glacier(gdir, candidates_df, ys)

        #plt.show()
