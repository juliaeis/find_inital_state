import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os

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
    print(surface_t0.head())

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
    #ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:',linewidth=2)

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
    plt.close()
    return

def plot_surface(gdir, plot_dir,i):

    plot_dir = os.path.join(plot_dir, 'surface')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    reconstruction = gdir.read_pickle('reconstruction_output')
    experiment = gdir.read_pickle('synthetic_experiment')

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
    ax1.annotate('t = 1865', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=18)
    ax2.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',
                 fontsize=13)

    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:',linewidth=3, label='solution' )
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k', linewidth=3,label='bed')
    ax1.plot(x, surface_t0.median(axis=0),linewidth=3, label='median')
    ax1.plot(x, experiment['y_t0'].fls[i].surface_h, 'k:',linewidth=3)
    ax1.plot(x, experiment['y_t0'].fls[i].bed_h, 'k',linewidth=3)

    ax1.fill_between(x, surface_t0.quantile(q=0.75, axis=0).values,
                     surface_t0.quantile(q=0.25, axis=0).values,
                     alpha=0.5,label='IQR')
    ax1.fill_between(x, surface_t0.min(axis=0).values,
                     surface_t0.max(axis=0).values, alpha=0.2, color='grey',
                     label='total range')

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

    plt.savefig(os.path.join(plot_dir,'surface.png'))
    plt.show()
    #plt.close()

    return

