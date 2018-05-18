import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial

from pylab import *
import os

from oggm import cfg, workflow, tasks, utils
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
from oggm.graphics import plot_catchment_width, plot_centerlines
FlowlineModel = partial(FluxBasedModel, inplace=False)


def make_result_panda(gdir,i):

    surface_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()
    widths_t0 = pd.DataFrame()
    widths_t = pd.DataFrame()

    reconstruction = gdir.read_pickle('reconstruction_output')
    for rec in reconstruction:
        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[i].surface_h],
                                               ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[i].surface_h],
                                               ignore_index=True)
            # widths
            widths_t = widths_t.append([rec[1].fls[i].widths],
                                             ignore_index=True)
            widths_t0 = widths_t0.append([rec[0].fls[i].widths],
                                             ignore_index=True)



def plot_climate(gdir, plot_dir) :
    #plot_dir = os.path.join(plot_dir, 'surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    d = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    temp = d.temp.resample(freq='12MS', dim='time', how=np.mean).to_series()
    temp = temp[temp.index.year >= 1850]

    fig, ax1 = plt.subplots(figsize=(15, 10))
    del temp.index.name
    temp.plot( linewidth=3, label='Annual temp')
    temp.rolling(31, center=True, min_periods=15).mean().plot(linewidth=3,
        label='31-yr avg')
    ax1.legend(loc='best', fontsize=25)
    plt.title('HISTALP annual temperature, Hintereisferner', fontsize=25)
    plt.ylabel(r'degC', fontsize=25)
    plt.xlabel(r'Time', fontsize=25)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'Histalps.pdf'), dpi=300)
    plt.show()

    return

def plot_experiment (gdir, plot_dir):
    plot_dir = os.path.join(plot_dir,'experiment')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    experiment = gdir.read_pickle('synthetic_experiment')
    fls_num = len(experiment['y_t0'].fls)-1
    x = np.arange(experiment['y_t'].fls[fls_num].nx) * experiment['y_t'].fls[
        fls_num].dx *  experiment['y_t'].fls[-1].map_dx
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': '+gdir.name,fontsize=25)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=30)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=30)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=25)

    ax1.plot(x, experiment['y_t0'].fls[fls_num].surface_h, 'k:', linewidth=3, label=r'$x_t$')
    ax1.plot(x, experiment['y_t0'].fls[fls_num].bed_h, 'k',linewidth=3, label=r'$b_t$ ')
    # ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:',linewidth=2)

    ax1.plot(x, experiment['y_t0'].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[fls_num].surface_h, 'k:')
    ax1.legend(loc='center left', bbox_to_anchor=(0.81, 1.1),fontsize=30)
    ax1.set_xlabel('Distance along the Flowline (m)',fontsize=35)
    ax1.set_ylabel('Altitude (m)',fontsize=35)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=22)
    ax2.set_ylabel('Altitude (m)',fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=25)

    #plt.savefig(os.path.join(plot_dir,'experiment_'+gdir.rgi_id+
    #                         '.pdf'),dpi=300)
    #plt.savefig(os.path.join(plot_dir,'experiment_' + gdir.rgi_id
    #                         + '.png'),dpi=300)
    plt.show()
    plt.close()
    return

def plot_surface(gdir, plot_dir,i):

    #plot_dir = os.path.join(plot_dir, 'surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    experiment = gdir.read_pickle('synthetic_experiment')
    fls = gdir.read_pickle('model_flowlines')

    surface_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()

    for rec in reconstruction:

        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[i].surface_h],
                                           ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[i].surface_h],
                                         ignore_index=True)

    x = np.arange(experiment['y_t'].fls[i].nx) * \
        experiment['y_t'].fls[i].dx * experiment['y_t'].fls[-1].map_dx

    fig, ax1 = plt.subplots(figsize=(25, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': '+gdir.name,fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=25)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=30)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=25)
    '''
    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:',linewidth=3, label=r'$\widetilde{x}_t$' )
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k', linewidth=3,label=r'$b_t$')
    ax1.plot(x, surface_t0.median(axis=0),linewidth=3, label='median'+r'$\left(\widehat{x}_t^j\right)$')
    '''
    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:', linewidth=3,
             label=r'$x_t$')
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k', linewidth=3,
             label=r'$b_t$')
    ax1.plot(x, surface_t0.median(axis=0), linewidth=3,
             label='median')
    ax1.fill_between(x, surface_t0.quantile(q=0.75, axis=0).values,
                     surface_t0.quantile(q=0.25, axis=0).values,
                     alpha=0.5,
                     label='IQR')
    ax1.fill_between(x, surface_t0.min(axis=0).values,
                     surface_t0.max(axis=0).values, alpha=0.2, color='grey',
                     label='total range' )

    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:',linewidth=3)
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k',linewidth=3)
    '''
    ax1.fill_between(x, surface_t0.quantile(q=0.75, axis=0).values,
                     surface_t0.quantile(q=0.25, axis=0).values,
                     alpha=0.5,label='IQR'+r'$\left(\widehat{x}_t^j\right)$')
    ax1.fill_between(x, surface_t0.min(axis=0).values,
                     surface_t0.max(axis=0).values, alpha=0.2, color='grey',
                     label='range'+r'$\left(\widehat{x}_t^j\right)$')
    '''
    ax2.plot(x, surface_t.median(axis=0),linewidth=2)
    ax2.fill_between(x, surface_t.quantile(q=0.75, axis=0).values,
                     surface_t.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, surface_t.min(axis=0).values,
                     surface_t.max(axis=0).values, alpha=0.2,color='grey')

    ax2.plot(x, experiment['y_t'].fls[i].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[i].surface_h, 'k:',linewidth=2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
    ax1.set_xlabel('Distance along the Flowline',fontsize=30)
    ax1.set_ylabel('Altitude (m)',fontsize=30)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=22)
    ax2.set_ylabel('Altitude (m)',fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    #plt.savefig(os.path.join(plot_dir,'surface'+gdir.rgi_id+'.png'))
    #plt.savefig(os.path.join(plot_dir, 'surface_HEF.pdf'))
    plt.show()
    plt.close()

    return

def plot_each_solution(gdir, plot_dir,i, best=True):

    #plot_dir = os.path.join(plot_dir, 'surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    experiment = gdir.read_pickle('synthetic_experiment')
    fls = gdir.read_pickle('model_flowlines')

    surface_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()
    widths_t0 = pd.DataFrame()
    widths_t = pd.DataFrame()

    analysis = pd.DataFrame()

    fig, ax1 = plt.subplots(figsize=(25, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=25)

    x = np.arange(experiment['y_t'].fls[i].nx) * \
        experiment['y_t'].fls[i].dx * experiment['y_t'].fls[-1].map_dx
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=30)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=25)

    for rec in reconstruction:

        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[i].surface_h],
                                           ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[i].surface_h],
                                         ignore_index=True)
            widths_t0 = widths_t0.append([rec[0].fls[i].widths],
                                           ignore_index=True)
            widths_t = widths_t.append([rec[1].fls[i].widths],
                                         ignore_index=True)


            #ax1.plot(x,rec[0].fls[i].surface_h, color='grey',alpha=0.3)
            #ax2.plot(x, rec[1].fls[i].surface_h,color='grey',alpha=0.3)
    #ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:', linewidth=3,
    #         label=r'$x_t$')
    #ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k', linewidth=3,
    #         label=r'$b_t$')

    if best:
        # calculate objective
        diff_s = surface_t.subtract(experiment['y_t'].fls[-1].surface_h,axis=1)**2
        diff_w = widths_t.subtract(experiment['y_t'].fls[-1].widths,axis=1)**2
        objective = diff_s.sum(axis=1) + diff_w.sum(axis=1)
        min_id = objective.argmin(axis=0)
        ax1.plot(x,surface_t0.iloc[min_id].values, 'r',linewidth=3, label='best objective')
        ax2.plot(x, surface_t.iloc[min_id].values, 'r', linewidth=2,
                 label='best objective')
    ax1.plot(x, surface_t0.median(axis=0), label='median', linewidth=3)

    ax1.fill_between(x, surface_t0.quantile(q=0.75, axis=0).values,
                     surface_t0.quantile(q=0.25, axis=0).values,
                     alpha=0.5, label='IQR')

    ax1.fill_between(x, surface_t0.min(axis=0).values,
                     surface_t0.max(axis=0).values, alpha=0.2, color='grey',
                     label='total range')

    ax2.plot(x, surface_t.median(axis=0), linewidth=2)
    ax2.fill_between(x, surface_t.quantile(q=0.75, axis=0).values,
                     surface_t.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, surface_t.min(axis=0).values,
                     surface_t.max(axis=0).values, alpha=0.2, color='grey')


    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:', linewidth=3)
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k', linewidth=3)
    ax2.plot(x, experiment['y_t'].fls[i].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[i].surface_h, 'k:',linewidth=2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
    ax1.set_xlabel('Distance along the Flowline',fontsize=30)
    ax1.set_ylabel('Altitude (m)',fontsize=30)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=22)
    ax2.set_ylabel('Altitude (m)',fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(os.path.join(plot_dir, 'median_'+str(gdir.rgi_id)+'.png'))
    #plt.show()
    plt.close()

    return

def plot_objective_surface(gdir, plot_dir,i, best=True):

    #plot_dir = os.path.join(plot_dir, 'surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    experiment = gdir.read_pickle('synthetic_experiment')
    fls = gdir.read_pickle('model_flowlines')

    surface_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()
    widths_t0 = pd.DataFrame()
    widths_t = pd.DataFrame()

    analysis = pd.DataFrame()

    plt.figure(figsize=(25,15))
    if gdir.name != "":
        plt.title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        plt.title(gdir.rgi_id, fontsize=25)

    x = np.arange(experiment['y_t'].fls[i].nx) * \
        experiment['y_t'].fls[i].dx * experiment['y_t'].fls[-1].map_dx

    plt.annotate(r'$t =  2000$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=30)


    for rec in reconstruction:

        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[i].surface_h],
                                           ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[i].surface_h],
                                         ignore_index=True)
            widths_t0 = widths_t0.append([rec[0].fls[i].widths],
                                           ignore_index=True)
            widths_t = widths_t.append([rec[1].fls[i].widths],
                                         ignore_index=True)

            plt.plot(x,rec[1].fls[i].surface_h-experiment['y_t'].fls[-1].surface_h, color='grey',alpha=0.3)

    if best:
        # calculate objective
        diff_s = surface_t.subtract(experiment['y_t'].fls[-1].surface_h,axis=1)**2
        diff_w = widths_t.subtract(experiment['y_t'].fls[-1].widths,axis=1)**2
        objective = diff_s.sum(axis=1) + diff_w.sum(axis=1)
        min_id = objective.argmin(axis=0)
        plt.plot(x,surface_t.iloc[min_id].values-experiment['y_t'].fls[-1].surface_h, 'r',linewidth=3, label='best objective')

    # plot median
    fls[-1].surface_h = surface_t0.median(axis=0).values
    past_climate = PastMassBalance(gdir)
    model=FluxBasedModel(fls,mb_model=past_climate,y0=1865)
    model.run_until(2000)
    plt.plot(x,model.fls[-1].surface_h-experiment['y_t'].fls[-1].surface_h, linewidth=3, label='median')

    '''
    ax1.plot(x, surface_t0.median(axis=0), linewidth=2)
    ax1.fill_between(x, surface_t0.quantile(q=0.75, axis=0).values,
                     surface_t0.quantile(q=0.25, axis=0).values,
                     alpha=0.5,label='IQR'+r'$\left(\widehat{x}_t^j\right)$')
    ax1.fill_between(x, surface_t0.min(axis=0).values,
                     surface_t0.max(axis=0).values, alpha=0.2, color='grey',
                     label='range'+r'$\left(\widehat{x}_t^j\right)$')

    ax2.plot(x, surface_t.median(axis=0),linewidth=2)
    ax2.fill_between(x, surface_t.quantile(q=0.75, axis=0).values,
                     surface_t.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, surface_t.min(axis=0).values,
                     surface_t.max(axis=0).values, alpha=0.2,color='grey')
    '''
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=15)
    plt.xlabel('Distance along the Flowline (m)',fontsize=30)
    plt.ylabel('Difference in Surface Elevation (m)',fontsize=30)

    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(os.path.join(plot_dir, 'diff_s_HEF.png'))
    plt.show()
    #plt.close()

    return

def plot_length(gdir,plot_dir):
    reconstruction = gdir.read_pickle('reconstruction_output')
    experiment = gdir.read_pickle('synthetic_experiment')
    surface_t0 = pd.DataFrame()
    widths_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()
    widths_t = pd.DataFrame()
    fls_t0 = pd.DataFrame()
    for rec in reconstruction:

        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[-1].surface_h],ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[-1].surface_h],ignore_index=True)
            widths_t0 = widths_t0.append([rec[0].fls[-1].widths],ignore_index=True)
            widths_t = widths_t.append([rec[1].fls[-1].widths],ignore_index=True)
            fls_t0 = fls_t0.append({'model':rec[0],'length':rec[0].length_m},ignore_index=True)

    past_climate = PastMassBalance(gdir)

    plt.figure(figsize=(25,15))

    for i in np.arange(0,30,1):
        fls = gdir.read_pickle('model_flowlines')
        try:
            fls[-1].surface_h = surface_t0.iloc[i].values

            past_model = FluxBasedModel(fls, mb_model=past_climate,
                                        glen_a=cfg.A, y0=1865)
            a, b = past_model.run_until_and_store(2000)
            plt.plot(b.length_m.to_series().rolling(36, center=True).mean()-b.length_m.to_series().iloc[-1], alpha=0.3,color='grey', label='')
        except:
            pass

    # mean plot
    fls = gdir.read_pickle('model_flowlines')
    fls[-1].surface_h = surface_t0.median(axis=0).values

    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)
    a,b = past_model.run_until_and_store(2000)
    plt.plot(b.length_m.to_series().rolling(36, center=True).mean()-b.length_m.to_series().iloc[-1],linewidth=3,  label = 'median')

    #objective plot

    # calculate objective
    diff_s = surface_t.subtract(experiment['y_t'].fls[-1].surface_h,axis=1)**2
    diff_w = widths_t.subtract(experiment['y_t'].fls[-1].widths,axis=1)**2
    objective = diff_s.sum(axis=1) + diff_w.sum(axis=1)
    min_id = objective.argmin(axis=0)

    fls = gdir.read_pickle('model_flowlines')
    fls[-1].surface_h = surface_t0.iloc[min_id].values

    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)
    a, b = past_model.run_until_and_store(2000)
    plt.plot(b.length_m.to_series().rolling(36, center=True).mean() -
             b.length_m.to_series().iloc[-1],'r', linewidth=3, label='best objective')

    # experiment plot
    fls = gdir.read_pickle('synthetic_experiment')['y_t0'].fls
    past_climate = PastMassBalance(gdir)
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)
    a, b = past_model.run_until_and_store(2000)
    plt.plot(b.length_m.to_series().rolling(36, center=True).mean() -
             b.length_m.to_series().iloc[-1], 'k:', linewidth=3,
             label='experiment')

    if gdir.name != "":
        plt.title(gdir.rgi_id + ': ' + gdir.name, fontsize=25)
    else:
        plt.title(gdir.rgi_id, fontsize=25)
    plt.ylabel('Glacier Lenght Change (m) ',fontsize=25)
    plt.xlabel('Time',fontsize=25)
    plt.legend(loc='best',fontsize=25)
    plt.xlim((1865,2000))
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig(os.path.join(plot_dir, 'lengths_RGI50-11-00687.pdf'), dpi=200)
    try:
        df = gdir.get_ref_length_data()
        df = df.loc[1855:2003]['dL']
        df = df - df.iloc[-1]
        plt.plot(df,'k',linewidth=3, label='real observations')

        fls = gdir.read_pickle('model_flowlines')
        past_model = FluxBasedModel(fls, mb_model=past_climate,
                                    glen_a=cfg.A, y0=1865)
        a, b = past_model.run_until_and_store(2000)
        plt.plot(b.length_m.to_series().rolling(36, center=True).mean() -
                 b.length_m.to_series().iloc[0],linewidth=3,  label='default initial state')
        plt.legend(loc='best',fontsize=25)
        plt.savefig(os.path.join(plot_dir, 'lengths_RGI50-11-00687_obs.pdf'), dpi=200)

    except:
        pass
    plt.show()
    return

def plot_issue(gdir,plot_dir):
    #plt.style.use('ggplot')

    workflow.gis_prepro_tasks([gdir])
    workflow.climate_tasks([gdir])
    workflow.inversion_tasks([gdir])
    tasks.init_present_time_glacier(gdir)

    # Observed length changes
    df = gdir.get_ref_length_data()
    df = df.loc[1855:2003]['dL']
    df = df - df.iloc[-1]


    tasks.run_from_climate_data(gdir,ys=1855,ye=2003,output_filesuffix='hist_from_current');
    ds = xr.open_dataset(
        gdir.get_filepath('model_diagnostics', filesuffix='hist_from_current'))
    (ds.length_m.to_series().rolling(36, center=True).mean()-ds.length_m.to_series().iloc[0]).plot(c='C0',label='OGGM')
    #s = s - s.iloc[-1]
    #print(s)
    ax = df.plot(c='k', label='Observations');
    #s.plot(c='C0', label='OGGM');
    plt.legend();
    ax.set_ylabel('Glacier Length Change [m]');
    plt.title('Hintereisferner length changes Experiment 2')
    plt.tight_layout();
    plt.show()
    '''
    fls = gdir.read_pickle('model_flowlines')
    x = np.arange(fls[-1].nx) *fls[-1].dx * fls[-1].map_dx

    plt.figure(figsize=(13,10))

    rc('axes', linewidth=3)

    plt.plot(x,fls[-1].surface_h,linewidth=3, label='Surface Elevation')
    plt.plot(x,fls[-1].bed_h,'k',linewidth=3,label='Bed Topography')
    plt.ylabel('Altitude (m)',size=30)
    plt.xlabel('Distance along the Flowline (m)',size=30)
    plt.legend(loc='best',fontsize=30)
    #plt.annotate('?', xy=(5000, 2700), fontsize=40)

    plt.tick_params(axis='both', which='major', labelsize=30)


    plt.title(gdir.rgi_id+ ': '+gdir.name,size=35)
    plt.savefig(os.path.join(plot_dir, 'issue_today.png'),dpi=200)
    '''
    #plt.savefig(os.path.join(plot_dir, 'issue_1850.pdf'),dpi=200)
    plt.show()

    return

