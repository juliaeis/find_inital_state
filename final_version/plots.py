import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
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
        ax1.set_title(gdir.rgi_id, fontsize=25)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=25)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.87), xycoords='axes fraction',
                 fontsize=20)

    ax1.plot(x, experiment['y_t0'].fls[fls_num].surface_h, 'k:', linewidth=3, label=r'$x_t$')
    ax1.plot(x, experiment['y_t0'].fls[fls_num].bed_h, 'k',linewidth=3, label=r'$b_t$ ')
    # ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:',linewidth=2)

    ax1.plot(x, experiment['y_t0'].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[fls_num].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[fls_num].surface_h, 'k:')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=25)
    ax1.set_xlabel('Distance along the Flowline (m)',fontsize=25)
    ax1.set_ylabel('Altitude (m)',fontsize=25)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=18)
    ax2.set_ylabel('Altitude (m)',fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(plot_dir,'experiment_'+gdir.rgi_id+
                             '.pdf'),dpi=300)
    plt.savefig(os.path.join(plot_dir,'experiment_' + gdir.rgi_id
                             + '.png'),dpi=300)
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

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': '+gdir.name,fontsize=25)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=25)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = 1865$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=25)
    ax2.annotate(r'$t = 2000$', xy=(0.15, 0.87), xycoords='axes fraction',
                 fontsize=20)

    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:',linewidth=3, label=r'$\widetilde{x}_t$' )
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k', linewidth=3,label=r'$b_t$')
    ax1.plot(x, surface_t0.median(axis=0),linewidth=3, label='median'+r'$\left(\widehat{x}_t^j\right)$')
    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:',linewidth=3)
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k',linewidth=3)

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

    ax2.plot(x, experiment['y_t'].fls[i].bed_h, 'k',linewidth=2)
    ax2.plot(x, experiment['y_t'].fls[i].surface_h, 'k:',linewidth=2)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=15)
    ax1.set_xlabel('Distance along the Flowline',fontsize=25)
    ax1.set_ylabel('Altitude (m)',fontsize=25)

    ax2.set_xlabel('Distance along the Flowline (m)',fontsize=18)
    ax2.set_ylabel('Altitude (m)',fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(plot_dir,'surface'+gdir.rgi_id+'.png'))
    plt.savefig(os.path.join(plot_dir, 'surface' + gdir.rgi_id + '.pdf'))
    plt.close()

    return

def plot_length(gdir,plot_dir):
    reconstruction = gdir.read_pickle('reconstruction_output')
    experiment = gdir.read_pickle('synthetic_experiment')
    surface_t0 = pd.DataFrame()
    widths_t0 = pd.DataFrame()
    fls_t0 = pd.DataFrame()
    for rec in reconstruction:

        if rec[0] != None:
            # surface
            surface_t0 = surface_t0.append([rec[0].fls[-1].surface_h],ignore_index=True)
            widths_t0 = widths_t0.append([rec[0].fls[-1].widths],ignore_index=True)
            fls_t0 = fls_t0.append({'model':rec[0],'length':rec[0].length_m},ignore_index=True)

    past_climate = PastMassBalance(gdir)

    plt.figure()

    for i in np.arange(0,4,1):
        fls = gdir.read_pickle('model_flowlines')
        fls[-1].surface_h = surface_t0.iloc[i].values

        past_model = FluxBasedModel(fls, mb_model=past_climate,
                                    glen_a=cfg.A, y0=1865)
        a, b = past_model.run_until_and_store(2000)
        plt.plot(b.length_m.to_series().rolling(36, center=True).mean(), color='grey')


    fls = gdir.read_pickle('model_flowlines')
    fls[-1].surface_h = surface_t0.median(axis=0).values

    past_climate = PastMassBalance(gdir)
    past_model = FluxBasedModel(fls, mb_model=past_climate,
                                glen_a=cfg.A, y0=1865)
    a,b = past_model.run_until_and_store(2000)

    plt.plot(b.length_m.to_series().rolling(36, center=True).mean())
    plt.title('Observed length changes Hintereisferner (1855-2003 )');
    plt.show()

    return

def plot_issue(gdir,plot_dir):
    #plt.style.use('ggplot')

    fls = gdir.read_pickle('model_flowlines')
    x = np.arange(fls[-1].nx) *fls[-1].dx * fls[-1].map_dx

    plt.figure(figsize=(13,10))


    plt.plot(x,fls[-1].surface_h,linewidth=3, label='Surface Elevation')
    plt.plot(x,fls[-1].bed_h,'k',linewidth=3,label='Bed Topography')
    plt.ylabel('Altitude (m)',size=30)
    plt.xlabel('Distance along the Flowline',size=30)
    plt.legend(loc='best',fontsize=30)


    plt.tick_params(axis='both', which='major', labelsize=30)


    plt.title(gdir.rgi_id+ ': '+gdir.name,size=35)
    plt.savefig(os.path.join(plot_dir, 'issue.png'),dpi=200)
    plt.savefig(os.path.join(plot_dir, 'issue.pdf'),dpi=200)
    plt.show()

    return

