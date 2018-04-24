import oggm
from oggm import cfg, workflow, tasks,graphics
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import LinearMassBalance, PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
from functools import partial
FlowlineModel = partial(FluxBasedModel, inplace=False)
import salem
import xarray as xr

import pickle
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import seaborn as sns
import copy

def diff(ar1,ar2):
    return abs(ar1-ar2)

def plot_surface (data_1880, data_2000,solution,rgi_id,fls_num, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if 'objective' in data_2000.columns:
        data_2000 = data_2000.drop('objective',axis=1)

    x = np.arange(solution[1].fls[fls_num].nx) * solution[1].fls[fls_num].dx * \
        solution[1].fls[-1].map_dx

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    ax1.set_title(rgi_id+': Hintereisferner',fontsize=25)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate('t = 1880', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=18)
    ax2.annotate('t = today', xy=(0.1, 0.9), xycoords='axes fraction',
                 fontsize=13)

    ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:',linewidth=3, label='solution' )
    ax1.plot(x, solution[0].fls[fls_num].bed_h, 'k', linewidth=3,label='bed')
    ax1.plot(x, data_1880.median(axis=0),linewidth=3, label='median')
    ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:',linewidth=3)
    ax1.plot(x, solution[0].fls[fls_num].bed_h, 'k',linewidth=3)

    ax1.fill_between(x, data_1880.quantile(q=0.75, axis=0).values,
                     data_1880.quantile(q=0.25, axis=0).values,
                     alpha=0.5,label='IQR')
    ax1.fill_between(x, data_1880.min(axis=0).values,
                     data_1880.max(axis=0).values, alpha=0.2, color='grey',
                     label='total range')

    ax2.plot(x, data_2000.median(axis=0),linewidth=2)
    ax2.fill_between(x, data_2000.quantile(q=0.75, axis=0).values,
                     data_2000.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, data_2000.min(axis=0).values,
                     data_2000.max(axis=0).values, alpha=0.2,color='grey')

    ax2.plot(x, solution[1].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, solution[1].fls[fls_num].surface_h, 'k:',linewidth=2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15)
    ax1.set_xlabel('Distance along the Flowline',fontsize=25)
    ax1.set_ylabel('Altitude (m)',fontsize=25)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=18)
    ax2.set_ylabel('Altitude (m)',fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(plot_dir,'surface.png'))
    #plt.close()
    return

def plot_experiment (solution,rgi_id,fls_num, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    x = np.arange(solution[1].fls[fls_num].nx) * solution[1].fls[fls_num].dx * \
        solution[1].fls[-1].map_dx

    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    ax1.set_title(rgi_id + ': Hintereisferner',fontsize=25)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=20)
    ax2.annotate(r'$t = 2000$', xy=(0.1, 0.87), xycoords='axes fraction',
                 fontsize=20)

    ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:', linewidth=3, label=r'$x_t$')
    ax1.plot(x, solution[0].fls[fls_num].bed_h, 'k',linewidth=3, label=r'$b_t$ ')
    #ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:',linewidth=2)

    ax1.plot(x, solution[0].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, solution[1].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, solution[1].fls[fls_num].surface_h, 'k:')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=25)
    ax1.set_xlabel('Distance along the Flowline (m)',fontsize=25)
    ax1.set_ylabel('Altitude (m)',fontsize=25)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=18)
    ax2.set_ylabel('Altitude (m)',fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    print(plot_dir)
    plt.savefig(os.path.join(plot_dir,'experiment_HEF.pdf'),dpi=300)
    #plt.close()
    return

def plot_problem(fls,rgi_id,fls_num, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    x = np.arange(fls[fls_num].nx) * fls[fls_num].dx * fls[-1].map_dx

    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_title(rgi_id + ': Hintereisferner',fontsize=25)

    ax1.plot(x, fls[fls_num].surface_h, linewidth=3, label='surface elevation')
    ax1.plot(x, fls[fls_num].bed_h, 'k',linewidth=3, label='bed topography')

    ax1.legend(loc='best',fontsize=15)
    ax1.set_xlabel('Distance along the Flowline (m)',fontsize=25)
    ax1.set_ylabel('Altitude (m)',fontsize=25)


    ax1.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(plot_dir,'main_flowline.png'),dpi=300)



    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_title(rgi_id + ': Hintereisferner', fontsize=25)

    ax1.plot(x, fls[fls_num].bed_h, 'k', linewidth=3, label='bed topography')
    ax1.annotate('?', xy=(5000, 2700), fontsize=40)
    ax1.legend(loc='best', fontsize=15)
    ax1.set_xlabel('Distance along the Flowline (m)', fontsize=25)
    ax1.set_ylabel('Altitude (m)', fontsize=25)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(plot_dir, 'flowline_1880.png'), dpi=300)
    #plt.close()
    return

# plot functions
def example_plot_temp_ts():
    d = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    temp = d.temp.resample(freq='12MS', dim='time', how=np.mean).to_series()
    temp = temp[temp.index.year>=1880]
    fig, ax1 = plt.subplots(figsize=(15, 10))
    del temp.index.name
    ax1.plot(temp,color='steelblue',linewidth=3, label='Annual temp')
    ax1.legend(loc='best', fontsize=25)
    plt.title('HISTALP annual temperature, Hintereisferner',fontsize=25)
    plt.ylabel(r'degC',fontsize=25)
    plt.xlabel(r'Time', fontsize=25)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Histalps.png'), dpi=300)


def plot_difference_histogramm(diff_s_1880,diff_s_2000,plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # plot histogramm difference

    f, ax = plt.subplots(2, 1,figsize=(13,10))
    main_fls_num = sorted(diff_w_1880.keys())[-1]
    ax[0].hist([diff_s_1880[i] for i in diff_s_1880.keys()],color='steelblue', bins=40)
    ax[1].hist([diff_s_2000[i] for i in diff_s_2000.keys()],color='steelblue', bins=40)

    ax[0].set_title(rgi_id+': Hintereisferner',fontsize=20)
    ax[0].set_ylabel('Frequency',size=20)
    ax[0].set_xlabel('Surface Error in 1880',size=20)

    ax[1].set_ylabel('Frequency',size=20)
    ax[1].set_xlabel('Surface Error today',size=20)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', which='major', labelsize=15)

    ax[0].legend(loc='best')
    ax[1].legend(loc='best')


    plt.savefig(os.path.join(plot_dir,'difference_histogramm.png'),dpi=300)

def plot_boxplot(volume,plot_dir):
    f,ax = plt.subplots(1,1,figsize=(11,15))
    ax.set_title(rgi_id + ': Hintereisferner', fontsize=25)
    sns.boxplot(data=volume)
    ax.set_xticklabels(['1880','today'], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_ylabel('Volume Difference (km³)',size=25)
    plt.savefig(os.path.join(plot_dir, 'boxplot_diff_volume.png'), dpi=300)

def plot_histogramm(volume,plot_dir):
    f,ax = plt.subplots(1,1,figsize=(10,15))
    ax.set_title(rgi_id + ': Hintereisferner', fontsize=25)
    ax.hist([volume['1880'],volume['today']],bins=30,label=['1880','today'])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('Frequency', size=25)
    ax.set_xlabel('Volume Difference (km³)',size=25)
    ax.legend(loc='best',fontsize=20)
    plt.savefig(os.path.join(plot_dir, 'hist_diff_volume.png'), dpi=300)

if __name__ == '__main__':

    path = '/home/juliaeis/Schreibtisch/cluster/initializing/run_100'
    plot_dir = '/home/juliaeis/Dokumente/Präsentationen/spp_Kickoff_2018/plots'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for file in os.listdir(path):


        s_1880 = {}
        s_2000 = {}
        w_1880 = {}
        w_2000 = {}

        diff_s_1880 = {}
        diff_s_2000 = {}
        diff_w_1880 = {}
        diff_w_2000 = {}


        if file.endswith('00897_solution.pkl') :

            rgi_id = file.split('_s')[0]
            solution_file = os.path.join(path,file)

            result_file = os.path.join(path,rgi_id+'.pkl')

            solution = pickle.load(open(solution_file,'rb'))
            results = pickle.load(open(result_file,'rb'))

            for i in range(len(solution[0].fls)):
                surface_1880 = pd.DataFrame()
                surface_2000 = pd.DataFrame()
                widths_1880 = pd.DataFrame()
                widths_2000 = pd.DataFrame()
                volume = pd.DataFrame()

                for res in results:
                    if res[0] != None:
                        # surface
                        surface_1880 = surface_1880.append([res[0].fls[i].surface_h],ignore_index=True)
                        surface_2000 = surface_2000.append([res[1].fls[i].surface_h],ignore_index=True)
                        # widths
                        widths_2000 = widths_2000.append([res[1].fls[i].widths], ignore_index=True)
                        widths_1880 = widths_1880.append([res[0].fls[i].widths],ignore_index=True)
                        if i == len(solution[0].fls)-1:
                            volume = volume.append({'1880':res[0].volume_km3-solution[0].volume_km3,'today':res[1].volume_km3-solution[1].volume_km3},ignore_index=True)


                s_1880[i] = copy.deepcopy(surface_1880)
                s_2000[i] = copy.deepcopy(surface_2000)
                w_1880[i] = copy.deepcopy(widths_1880)
                w_2000[i] = copy.deepcopy(widths_2000)


            # calculate objective
            sum_s = surface_2000.apply(diff,axis=1,args=[solution[1].fls[-1].surface_h])**2
            sum_w = widths_2000.apply(diff,axis=1,args=[solution[1].fls[-1].widths])**2
            objective = sum_s.sum(axis=1)+sum_w.sum(axis=1)
            surface_2000['objective'] = objective

            #filter results
            filtered = surface_2000['objective'] < surface_2000['objective'].median() + surface_2000['objective'].mad()

            for i in [2]:

                s_1880[i] = s_1880[i][filtered]
                s_2000[i] = s_2000[i][filtered]
                w_1880[i] = w_1880[i][filtered]
                w_2000[i] = w_2000[i][filtered]




                # calculate difference for all flowlines

                diff_s_1880[i] = s_1880[i].apply(diff, axis=1, args=[solution[0].fls[i].surface_h]).sum(axis=1)
                diff_s_2000[i] = s_2000[i].apply(diff, axis=1, args=[solution[1].fls[i].surface_h]).sum(axis=1)
                diff_w_1880[i] = w_1880[i].apply(diff, axis=1, args=[solution[0].fls[i].widths]).sum(axis=1)
                diff_w_2000[i] = w_2000[i].apply(diff, axis=1,args=[solution[1].fls[i].widths]).sum(axis=1)


            plot_experiment( solution,rgi_id,2, os.path.join(plot_dir))
            #plot_surface(s_1880[2], s_2000[2], solution, rgi_id,2, os.path.join(plot_dir))
            #plot_difference_histogramm(diff_s_1880, diff_s_2000, plot_dir)
            '''
            plot_boxplot(volume,plot_dir)
            plot_histogramm(volume, plot_dir)


            plot_difference_boxplot(diff_s_1880, diff_s_2000, diff_w_1880, diff_w_2000, os.path.join(plot_dir,'boxplot'))
            plot_difference_length(results,solution, rgi_id, plot_dir)


            cfg.initialize()
            cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
            cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
            cfg.PATHS[
                'working_dir'] = '/home/juliaeis/PycharmProjects/find_inital_state/test_HEF'
            # cfg.PATHS['working_dir'] = os.environ.get("S_WORKDIR")
            cfg.PARAMS['border'] = 10
            cfg.PARAMS['prcp_scaling_factor']
            cfg.PARAMS['run_mb_calibration'] = True
            cfg.PARAMS['optimize_inversion_params'] = True
            cfg.PARAMS['use_intersects'] = False
            plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size
            import geopandas as gpd
            rgi = get_demo_file('rgi_oetztal.shp')
            # rgi = get_demo_file('HEF_MajDivide.shp')

            hef = gpd.read_file(rgi)
            gdirs = workflow.init_glacier_regions(salem.read_shapefile(rgi))
            workflow.execute_entity_task(tasks.glacier_masks, gdirs,reset=True)
            for gdir in gdirs:
                if gdir.rgi_id =='RGI50-11.00897':

                    #plot_problem(gdir.read_pickle('model_flowlines'), rgi_id, i, os.path.join(plot_dir))
                    #example_plot_temp_ts()

                    fig, ax1 = plt.subplots(figsize=(6, 4))
                    graphics.plot_centerlines(gdir,ax=ax1)
                    #ax1.tick_params(axis='both', which='major', labelsize=20)
                    #ax1.set_title(gdir.rgi_id+': Hintereisferner',size=20)

            '''

        plt.show()