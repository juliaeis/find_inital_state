import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#import seaborn as sns
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial

from pylab import *
import os

import copy
from oggm import cfg, workflow, tasks, utils
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import PastMassBalance, RandomMassBalance
from oggm.core.flowline import FluxBasedModel
from oggm.graphics import plot_catchment_width, plot_centerlines
FlowlineModel = partial(FluxBasedModel, inplace=False)


def find_best_objective(gdir,y_te,t0,te):
    results = gdir.read_pickle('reconstruction_output')
    df = pd.DataFrame(columns=[str(t0),str(te)])
    for res in results:
        if res[0] != None:
            df = df.append(pd.DataFrame([res],columns=[str(t0),str(te)]),
                           ignore_index=True)
    df['objective'] = df[str(te)].apply((lambda x: objective_value(y_te,x.fls)))
    return df, df['objective'].idxmin()

def objective_value(fls1,fls2):
    return(np.sum(abs(fls1[-1].surface_h-fls2[-1].surface_h)**2)+ \
          np.sum(abs(fls1[-1].widths-fls2[-1].widths)**2))


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

def plot_surface(gdir, plot_dir,i,best=True,synthetic_exp=True):

    plot_dir = os.path.join(plot_dir, 'surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    fls = gdir.read_pickle('model_flowlines')

    surface_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()
    widths_t = pd.DataFrame()

    for rec in reconstruction:

        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[i].surface_h],
                                           ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[i].surface_h],
                                         ignore_index=True)
            widths_t = widths_t.append([rec[1].fls[i].widths],
                                           ignore_index=True)

    x = np.arange(fls[i].nx) * fls[i].dx * fls[i].map_dx


    fig, ax1 = plt.subplots(figsize=(27, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': '+gdir.name,fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=30)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=35)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=20)

    ax1.plot(x, fls[i].bed_h, 'k', linewidth=3,
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

    ax2.plot(x, surface_t.median(axis=0), linewidth=2)
    ax2.fill_between(x, surface_t.quantile(q=0.75, axis=0).values,
                     surface_t.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, surface_t.min(axis=0).values,
                     surface_t.max(axis=0).values, alpha=0.2, color='grey')

    if synthetic_exp:
        experiment = gdir.read_pickle('synthetic_experiment')
        ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:', linewidth=3)
        ax2.plot(x, experiment['y_t'].fls[i].surface_h, 'k:', linewidth=2)
    else:
        ax2.plot(x, fls[i].surface_h, 'k', linewidth=2)

    if best:
        if synthetic_exp:
            # calculate objective
            df, min_id = find_best_objective(gdir,experiment['y_t'].fls,1865,2000)
        else:
            df, min_id = find_best_objective(gdir, fls, 1865, 2000)
        ax1.plot(x,surface_t0.iloc[min_id].values, 'r',linewidth=3, label='best objective')
        ax2.plot(x, surface_t.iloc[min_id].values, 'r', linewidth=2,
                 label='best objective')

    ax1.plot(x, fls[i].bed_h, 'k', linewidth=3)
    ax2.plot(x, fls[i].bed_h, 'k', linewidth=3)


    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=25)
    ax1.set_xlabel('Distance along the Flowline',fontsize=35)
    ax1.set_ylabel('Altitude (m)',fontsize=35)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=25)
    ax2.set_ylabel('Altitude (m)',fontsize=25)
    ax1.tick_params(axis='both', which='major', labelsize=35)
    ax2.tick_params(axis='both', which='major', labelsize=30)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_linewidth(1.5)

    plt.savefig(os.path.join(plot_dir,'surface'+gdir.rgi_id+'.png'),dpi=200)
    #plt.show()
    plt.close()

    return

def plot_each_surface(gdir, plot_dir,i, best=True, synthetic_exp=True):

    plot_dir = os.path.join(plot_dir, 'individual_surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    fls = gdir.read_pickle('model_flowlines')

    surface_t0 = pd.DataFrame()
    surface_t = pd.DataFrame()

    fig, ax1 = plt.subplots(figsize=(25, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=25)

    x = np.arange(fls[i].nx) *fls[i].dx *fls[i].map_dx

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

            ax1.plot(x,rec[0].fls[i].surface_h, color='grey',alpha=0.3)
            ax2.plot(x, rec[1].fls[i].surface_h,color='grey',alpha=0.3)

    if synthetic_exp:
        experiment = gdir.read_pickle('synthetic_experiment')
        ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:', linewidth=3)
        ax2.plot(x, experiment['y_t'].fls[i].surface_h, 'k:', linewidth=2)
    else:
        ax2.plot(x, fls[i].surface_h, 'k', linewidth=2)

    if best:
        if synthetic_exp:
            # calculate objective
            df, min_id = find_best_objective(gdir, experiment['y_t'].fls, 1865,2000)
        else:
            df, min_id = find_best_objective(gdir, fls, 1865, 2000)

        ax1.plot(x,df.loc[min_id,'1865'].fls[i].surface_h, 'r',linewidth=3, label='best objective')
        ax2.plot(x, df.loc[min_id,'2000'].fls[i].surface_h, 'r', linewidth=2,
                 label='best objective')

    ax1.plot(x, fls[i].bed_h, 'k', linewidth=3)
    ax2.plot(x, fls[i].bed_h, 'k', linewidth=2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
    ax1.set_xlabel('Distance along the Flowline',fontsize=30)
    ax1.set_ylabel('Altitude (m)',fontsize=30)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=22)
    ax2.set_ylabel('Altitude (m)',fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(os.path.join(plot_dir, 'individual_surface'+str(gdir.rgi_id)+'.png'))
    #plt.show()
    plt.close()

    return


def plot_widths(gdir, plot_dir,i,best=True,synthetic_exp=True):

    plot_dir = os.path.join(plot_dir, 'widths')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    fls = gdir.read_pickle('model_flowlines')

    surface_t = pd.DataFrame()
    widths_t0 = pd.DataFrame()
    widths_t = pd.DataFrame()

    for rec in reconstruction:

        if rec[0] != None:
            # surface
            widths_t0 = widths_t0.append([rec[0].fls[i].widths],
                                           ignore_index=True)
            surface_t = surface_t.append([rec[1].fls[i].surface_h],
                                         ignore_index=True)
            widths_t = widths_t.append([rec[1].fls[i].widths],
                                           ignore_index=True)

    x = np.arange(fls[i].nx) * fls[i].dx * fls[i].map_dx


    fig, ax1 = plt.subplots(figsize=(27, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': '+gdir.name,fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=30)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=35)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=20)

    ax1.plot(x, widths_t0.median(axis=0), linewidth=3,
             label='median')
    ax1.fill_between(x, widths_t0.quantile(q=0.75, axis=0).values,
                     widths_t0.quantile(q=0.25, axis=0).values,
                     alpha=0.5,
                     label='IQR')
    ax1.fill_between(x, widths_t0.min(axis=0).values,
                     widths_t0.max(axis=0).values, alpha=0.2, color='grey',
                     label='total range' )

    ax2.plot(x, widths_t.median(axis=0), linewidth=2)
    ax2.fill_between(x, widths_t.quantile(q=0.75, axis=0).values,
                     widths_t.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, widths_t.min(axis=0).values,
                     widths_t.max(axis=0).values, alpha=0.2, color='grey')

    if synthetic_exp:
        experiment = gdir.read_pickle('synthetic_experiment')
        ax1.plot(x, experiment['y_t0'].fls[i].widths, 'k:', linewidth=3)
        ax2.plot(x, experiment['y_t'].fls[i].widths, 'k:', linewidth=2)
    else:
        ax2.plot(x, fls[i].widths, 'k', linewidth=2)

    if best:
        if synthetic_exp:
            # calculate objective
            df, min_id = find_best_objective(gdir,experiment['y_t'].fls,1865,2000)
        else:
            df, min_id = find_best_objective(gdir, fls, 1865, 2000)

        ax1.plot(x,widths_t0.iloc[min_id].values, 'r',linewidth=3, label='best objective')
        ax2.plot(x, widths_t.iloc[min_id].values, 'r', linewidth=2,
                 label='best objective')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=25)
    ax1.set_xlabel('Distance along the Flowline',fontsize=35)
    ax1.set_ylabel('Altitude (m)',fontsize=35)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=25)
    ax2.set_ylabel('Altitude (m)',fontsize=25)
    ax1.tick_params(axis='both', which='major', labelsize=35)
    ax2.tick_params(axis='both', which='major', labelsize=30)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax2.spines[axis].set_linewidth(1.5)

    plt.savefig(os.path.join(plot_dir,'widths'+gdir.rgi_id+'.png'),dpi=200)
    #plt.show()
    plt.close()

    return

def plot_each_widths(gdir, plot_dir,i, best=True, synthetic_exp=True):

    plot_dir = os.path.join(plot_dir, 'individual_widths')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    fls = gdir.read_pickle('model_flowlines')

    widths_t0 = pd.DataFrame()
    widths_t = pd.DataFrame()

    fig, ax1 = plt.subplots(figsize=(25, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=25)

    x = np.arange(fls[i].nx) *fls[i].dx *fls[i].map_dx

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=30)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=25)

    for rec in reconstruction:

        if rec[0] != None:
            # surface
            widths_t0 = widths_t0.append([rec[0].fls[i].surface_h],
                                           ignore_index=True)
            widths_t = widths_t.append([rec[1].fls[i].surface_h],
                                         ignore_index=True)

            ax1.plot(x,rec[0].fls[i].widths, color='grey',alpha=0.3)
            ax2.plot(x, rec[1].fls[i].widths,color='grey',alpha=0.3)

    if synthetic_exp:
        experiment = gdir.read_pickle('synthetic_experiment')
        ax1.plot(x, experiment['y_t0'].fls[i].widths, 'k:', linewidth=3)
        ax2.plot(x, experiment['y_t'].fls[i].widths, 'k:', linewidth=2)
    else:
        ax2.plot(x, fls[i].widths, 'k', linewidth=2)

    if best:
        if synthetic_exp:
            # calculate objective
            df, min_id = find_best_objective(gdir, experiment['y_t'].fls, 1865,2000)
        else:
            df, min_id = find_best_objective(gdir, fls, 1865, 2000)

        ax1.plot(x,df.loc[min_id,'1865'].fls[i].widths, 'r',linewidth=3, label='best objective')
        ax2.plot(x, df.loc[min_id,'2000'].fls[i].widths, 'r', linewidth=2,
                 label='best objective')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=20)
    ax1.set_xlabel('Distance along the Flowline',fontsize=30)
    ax1.set_ylabel('Altitude (m)',fontsize=30)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=22)
    ax2.set_ylabel('Altitude (m)',fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig(os.path.join(plot_dir, 'individual_widths'+str(gdir.rgi_id)+'.png'))
    #plt.show()
    plt.close()

    return


def plot_length_only(gdir,plot_dir,t0,te,synthetic_exp=True):
    plot_dir = os.path.join(plot_dir, 'length_only')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    reconstruction = gdir.read_pickle('reconstruction_output')
    reconstruction = [x for x in reconstruction if x[0] is not None]
    surface_t0 = pd.DataFrame()
    for rec in reconstruction:
        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[-1].surface_h],ignore_index=True)

    past_climate = PastMassBalance(gdir)

    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot(111)

    for i in np.arange(0,50,1):
        #fls = gdir.read_pickle('model_flowlines')
        if reconstruction[i][0] != None:
            fls= copy.deepcopy(reconstruction[i][0].fls)
            past_model = FluxBasedModel(fls, mb_model=past_climate,
                                        glen_a=cfg.A, y0=t0)
            try:
                a, b = past_model.run_until_and_store(te)
                (b.length_m.rolling(time=36,center=True).mean()).plot(ax=ax, alpha=0.3,color='grey', label='')
            except:
                pass


    # median plot
    fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    fls[-1].surface_h = surface_t0.median(axis=0).values

    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    a,b = past_model.run_until_and_store(te)
    (b.length_m.rolling(time=36, center=True).mean()).plot(ax=ax,linewidth=3,  label = 'median')

    #objective plot

    # calculate objective
    if synthetic_exp:
        experiment = gdir.read_pickle('synthetic_experiment')
        df,min_id = find_best_objective(gdir,experiment['y_t'].fls,1865,2000)
    else:
        df,min_id = find_best_objective(gdir, fls, 1865, 2000)

    fls = copy.deepcopy(df.loc[min_id,'1865'].fls)
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    a, b = past_model.run_until_and_store(te)
    (b.length_m.rolling(time=36, center=True).mean()).plot(ax=ax,color='red', linewidth=3, label='best objective')

    fls = gdir.read_pickle('model_flowlines')
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    plt.plot(2000,past_model.length_m,'ko', markersize=12)

    if synthetic_exp:
        # experiment plot
        fls = copy.deepcopy(gdir.read_pickle('synthetic_experiment')['y_t0'].fls)
        past_climate = PastMassBalance(gdir)
        past_model = FluxBasedModel(fls, mb_model=past_climate,
                                    glen_a=cfg.A, y0=t0)
        a, b = past_model.run_until_and_store(te)
        (b.length_m.rolling(time=36, center=True).mean()).plot(ax=ax,color='k',linestyle=':', linewidth=3,
                 label='experiment')
    else:
        try:
            df = gdir.get_ref_length_data()
            df = df.loc[1855:2000]['dL']
            df = df - df.iloc[-1]+past_model.length_m
            df = df.reset_index().set_index('years')
            df = df.rename(columns={'dL':'real observations'})
            df.plot(ax=ax,use_index=True,color='k', linewidth=3, label='real observations')
        except:
           pass


    a, b = past_model.run_until_and_store(te)
    (b.length_m.rolling(time=36, center=True).mean()).plot(ax=ax,linewidth=3,  label='default initial state')
    plt.legend(loc='best',fontsize=30)

    if gdir.name != "":
        ax.set_title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        ax.set_title(gdir.rgi_id, fontsize=30)

    ax.set_xlabel('Time', fontsize=35)
    ax.legend(loc='best', fontsize=30)
    plt.xlim((t0, te))
    plt.tick_params(axis='both', which='major', labelsize=35)
    ax.set_ylabel('Glacier Length (m) ', fontsize=35)
    plt.savefig(os.path.join(plot_dir,'lengths_'+str(gdir.rgi_id)+'.png'), dpi=300)

    #plt.show()
    return


def plot_length_change(gdir,plot_dir,t0,te,synthetic_exp=True):
    plot_dir = os.path.join(plot_dir, 'length_change')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    reconstruction = gdir.read_pickle('reconstruction_output')

    surface_t0 = pd.DataFrame()
    for rec in reconstruction:
        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[-1].surface_h],ignore_index=True)

    past_climate = PastMassBalance(gdir)

    fig = plt.figure(figsize=(25,15))
    ax = fig.add_subplot(111)

    for i in np.arange(0,5,1):
        #fls = gdir.read_pickle('model_flowlines')
        if reconstruction[i][0] != None:
            fls= copy.deepcopy(reconstruction[i][0].fls)
            past_model = FluxBasedModel(fls, mb_model=past_climate,
                                        glen_a=cfg.A, y0=t0)
            a, b = past_model.run_until_and_store(te)
            (b.length_m.rolling(time=36,center=True).mean()-b.length_m[-1]).plot(ax=ax, alpha=0.3,color='grey', label='')


    # median plot
    fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    fls[-1].surface_h = surface_t0.median(axis=0).values

    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    a,b = past_model.run_until_and_store(te)
    (b.length_m.rolling(time=36, center=True).mean()-b.length_m[-1]).plot(ax=ax,linewidth=3,  label = 'median')

    #objective plot

    # calculate objective
    if synthetic_exp:
        experiment = gdir.read_pickle('synthetic_experiment')
        df,min_id = find_best_objective(gdir,experiment['y_t'].fls,1865,2000)
    else:
        df,min_id = find_best_objective(gdir, fls, 1865, 2000)

    fls = copy.deepcopy(df.loc[min_id,'1865'].fls)
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    a, b = past_model.run_until_and_store(te)
    (b.length_m.rolling(time=36, center=True).mean()-b.length_m[-1]).plot(ax=ax,color='red', linewidth=3, label='best objective')

    fls = gdir.read_pickle('model_flowlines')
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)

    if synthetic_exp:
        # experiment plot
        fls = copy.deepcopy(gdir.read_pickle('synthetic_experiment')['y_t0'].fls)
        past_climate = PastMassBalance(gdir)
        past_model = FluxBasedModel(fls, mb_model=past_climate,
                                    glen_a=cfg.A, y0=t0)
        a, b = past_model.run_until_and_store(te)
        (b.length_m.rolling(time=36, center=True).mean()-b.length_m[-1]).plot(ax=ax,color='k',linestyle=':', linewidth=3,
                 label='experiment')
    else:

        try:
            df = gdir.get_ref_length_data()
            df = df.loc[1855:2000]['dL']
            df = df - df.iloc[-1]
            df = df.reset_index().set_index('years')
            df = df.rename(columns={'dL':'real observations'})
            df.plot(ax=ax,use_index=True,color='k', linewidth=3, label='real observations')
        except:
           pass

    fls = gdir.read_pickle('model_flowlines')
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)


    fls = gdir.read_pickle('model_flowlines')
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=t0)
    a, b = past_model.run_until_and_store(te)
    (b.length_m.rolling(time=36, center=True).mean()-b.length_m[-1]).plot(ax=ax,linewidth=3,  label='default initial state')
    plt.legend(loc='best',fontsize=30)
    if gdir.name != "":
        ax.set_title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        ax.set_title(gdir.rgi_id, fontsize=30)

    ax.set_xlabel('Time', fontsize=35)
    ax.legend(loc='best', fontsize=30)
    plt.xlim((t0, te))
    plt.tick_params(axis='both', which='major', labelsize=35)
    ax.set_ylabel('Glacier Length Change (m) ', fontsize=35)
    plt.savefig(os.path.join(plot_dir, 'lengths_change_'+str(gdir.rgi_id)+'.png'), dpi=300)

    return


def test_animation(gdir):

    import matplotlib.pyplot as plt
    from matplotlib import animation
    mpl.style.use('default')


    fig = plt.figure(figsize=(20,15))
    ax1 = plt.axes()
    fill = ax1.fill_between([], [],color='grey',alpha=0.1,label='total range', lw=2)
    fill2 = ax1.fill_between([], [], color='C0', alpha=0.5,label='IQR', lw=2)
    time_text = ax1.text(0.6, 0.95, '', transform=ax1.transAxes, size=25)


    plotlays, plotcols, label  = [2], ["orange","red","C0"] , ['default', 'best objective', 'median']
    lines = []
    for index in range(3):
        lobj = ax1.plot([],[],lw=2,color=plotcols[index],label=label[index])[0]
        lines.append(lobj)

    fls = gdir.read_pickle('model_flowlines')
    past_climate = PastMassBalance(gdir)
    past_model = FluxBasedModel(copy.deepcopy(fls), mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)
    # best objective model
    experiment = gdir.read_pickle('synthetic_experiment')
    df,best = find_best_objective(gdir,experiment['y_t'].fls,1865,2000)
    best_fls = copy.deepcopy(df.loc[best,'1865'].fls)
    best_model = FluxBasedModel(copy.deepcopy(best_fls), mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)


    surface = pd.DataFrame()
    for i in df.index:
        surface = surface.append([df.loc[i,'1865'].fls[-1].surface_h],ignore_index=True)

    # median model
    median_fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    median_fls[-1].surface_h = copy.deepcopy(surface.median(axis=0).values)

    median_model = FluxBasedModel(copy.deepcopy(median_fls), mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)

    # quant_25 model
    quant25_fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    quant25_fls[-1].surface_h = copy.deepcopy(surface.quantile(q=0.25, axis=0).values)

    quant25_model = FluxBasedModel(copy.deepcopy(quant25_fls),
                                  mb_model=past_climate,
                                  glen_a=cfg.A, y0=1865)

    # quant_75 model
    quant75_fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    quant75_fls[-1].surface_h = copy.deepcopy(
        surface.quantile(q=0.75, axis=0).values)

    quant75_model = FluxBasedModel(copy.deepcopy(quant75_fls),
                                   mb_model=past_climate,
                                   glen_a=cfg.A, y0=1865)

    #min
    min_fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    min_fls[-1].surface_h = copy.deepcopy(surface.min(axis=0).values)

    min_model = FluxBasedModel(copy.deepcopy(min_fls),
                                  mb_model=past_climate,
                                  glen_a=cfg.A, y0=1865)

    #max
    max_fls = copy.deepcopy(gdir.read_pickle('model_flowlines'))
    max_fls[-1].surface_h = copy.deepcopy(surface.max(axis=0).values)

    max_model = FluxBasedModel(copy.deepcopy(max_fls),
                                  mb_model=past_climate,
                                  glen_a=cfg.A, y0=1865)

    x = np.arange(fls[-1].nx) * fls[-1].dx * fls[-1].map_dx

    def init():
        ax1.plot(x, fls[-1].bed_h, 'k')
        time_text.set_text('')
        for line in lines:
            line.set_data([],[])
        return lines



    def animate(t):
        if t ==1865:
            fls = gdir.read_pickle('model_flowlines')
            past_model.reset_flowlines(copy.deepcopy(fls))
            past_model.reset_y0(1865)

            best_model.reset_flowlines(copy.deepcopy(df.loc[best,'1865'].fls))
            best_model.reset_y0(1865)

            fls = gdir.read_pickle('model_flowlines')
            fls[-1].surface_h = surface.median(axis=0).values
            median_model.reset_flowlines(copy.deepcopy(fls))
            median_model.reset_y0(1865)

            fls = gdir.read_pickle('model_flowlines')
            fls[-1].surface_h = surface.quantile(0.25,axis=0).values
            quant25_model.reset_flowlines(copy.deepcopy(fls))
            quant25_model.reset_y0(1865)

            fls = gdir.read_pickle('model_flowlines')
            fls[-1].surface_h = surface.quantile(0.75, axis=0).values
            quant75_model.reset_flowlines(copy.deepcopy(fls))
            quant75_model.reset_y0(1865)

            fls = gdir.read_pickle('model_flowlines')
            fls[-1].surface_h = surface.min(axis=0).values
            min_model.reset_flowlines(copy.deepcopy(fls))
            min_model.reset_y0(1865)

            fls = gdir.read_pickle('model_flowlines')
            fls[-1].surface_h = surface.max(axis=0).values
            max_model.reset_flowlines(copy.deepcopy(fls))
            max_model.reset_y0(1865)


        else:
            past_model.run_until(t)
            best_model.run_until(t)
            median_model.run_until(t)
            min_model.run_until(t)
            max_model.run_until(t)
            quant25_model.run_until(t)
            quant75_model.run_until(t)

        time_text.set_text('time = %.1f' % t)

        y1 = past_model.fls[-1].surface_h
        y2 = best_model.fls[-1].surface_h
        y3 = median_model.fls[-1].surface_h
        y4 = min_model.fls[-1].surface_h
        y5 = max_model.fls[-1].surface_h
        y6 = quant25_model.fls[-1].surface_h
        y7 = quant75_model.fls[-1].surface_h

        xlist = [x, x, x]
        ylist = [y1, y2, y3]
        ax1.collections.clear()
        fill = ax1.fill_between(x,y4,y5,color='grey',alpha=0.2,label='total range')
        fill2 = ax1.fill_between(x,y6,y7,color='C0',alpha=0.5, label = 'IQR')

        #for index in range(0,1):
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.

        return (fill2,)+tuple(lines)+(fill,)  + (time_text,)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    ani = animation.FuncAnimation(fig, animate, frames=range(1865, 2005, 5),
                        init_func=init, blit=True)

    plt.legend(loc='best',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel('Distance along the Flowline (m)',fontsize=25)
    plt.ylabel('Altitude (m)',fontsize=25)
    if gdir.name != "":
        plt.title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        plt.title(gdir.rgi_id, fontsize=30)
    ani.save(os.path.join(gdir.dir,'surface_animation.mp4'))

    #plt.show()
