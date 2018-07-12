from functools import partial
from pylab import *
from oggm.core.flowline import FluxBasedModel, FileModel
from oggm import graphics, tasks
from matplotlib import cm
import pandas as pd
from multiprocessing import Pool
from copy import deepcopy
FlowlineModel = partial(FluxBasedModel, inplace=False)

import os


def plot_candidates(gdir,df,experiment,ys):
    plot_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/past_state_information/plots'

    fig,ax = plt.subplots()
    # plot random run
    for suffix in df['suffix'].unique():
        rp = gdir.get_filepath('model_run',filesuffix=suffix)
        fmod = FileModel(rp)
        fmod.volume_m3_ts().plot(ax=ax,color='grey',label='',zorder=1)

    # last one again for labeling
    df['temp_bias'] = df['suffix'].apply(lambda x: float(x.split('_')[-1]))
    label = r'temperature bias $\in [$' + str(
        df['temp_bias'].min()) + ',' + str(df['temp_bias'].max()) + '$]$'
    fmod.volume_m3_ts().plot(ax=ax, color='grey', label=label,zorder=1)
    t_eq = df['time'].sort_values().iloc[0]
    ax.axvline(x=t_eq,color='k',zorder=1)

    df.plot.scatter(x='time',y='ts_section',ax=ax,c='objective',
                    colormap='RdYlGn_r',norm=mpl.colors.LogNorm(vmin=0.1, vmax=1e5),s=40,
                    zorder=2)


    plt.ylabel(r'Volume $(m^3)$')
    plt.title(gdir.rgi_id)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'random'+str(ys)+'_' + str(gdir.rgi_id) + '.png'),
                dpi=200)
    plt.close()
    #plt.show()


def plot_volume_dif_time(gdir,dict,experiment):

    fig, axs = plt.subplots(len(dict), 1)

    try:
        rp = gdir.get_filepath('model_run',filesuffix='experiment')
        ex_mod = FileModel(rp)
    except:
        # create volume plot from experiment
        model = experiment['y_t0']
        tasks.run_from_climate_data(gdir, ys=1850, ye=2000,
                                    init_model_fls=model.fls,
                                    output_filesuffix='experiment')
        rp = gdir.get_filepath('model_run', filesuffix='experiment')
        ex_mod = FileModel(rp)

    if gdir.name != '':
        plt.suptitle(gdir.rgi_id+':'+gdir.name,fontsize=20)
    else:
        plt.suptitle(gdir.rgi_id, fontsize=20)

    import matplotlib as mpl
    import matplotlib.cm as cm

    norm = mpl.colors.LogNorm(vmin=0.1, vmax=1e5)
    cmap = matplotlib.cm.get_cmap('RdYlGn_r')

    for i, ax in enumerate(fig.axes):
        yr = list(dict.keys())[i]
        df = dict.get(yr)
        df = df.sort_values('objective', ascending=False)
        for i, model in df['model_t0'].iteritems():
            color = cmap(norm(df.loc[i, 'objective']))
            model.volume_m3_ts().plot(ax=ax, color=color, linewidth=2)
        ex_mod.volume_m3_ts().plot(ax=ax, color='k', linestyle=':', linewidth=3)



def plot_surface_col(gdir,df,experiment,ys):
    df = df[df['objective']<=100]
    x = np.arange(experiment['y_t'].fls[-1].nx) * \
        experiment['y_t'].fls[-1].dx * experiment['y_t'].fls[-1].map_dx
    fig = plt.figure(figsize=(20,15))
    grid = plt.GridSpec(2, 2, hspace=0.2, wspace=0.2)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, :])

    p2 = ax2.get_position()

    if gdir.name != '':
        plt.suptitle(gdir.rgi_id+':'+gdir.name,x=p2.x1/2,fontsize=20)
    else:
        plt.suptitle(gdir.rgi_id, x=p2.x1/2, fontsize=20)

    import matplotlib as mpl
    import matplotlib.cm as cm

    norm = mpl.colors.LogNorm(vmin=0.1, vmax=1e5)
    cmap = matplotlib.cm.get_cmap('RdYlGn_r')
    df = df.sort_values('objective',ascending=False)
    for i,model in df['model_t0'].iteritems():
        color = cmap(norm(df.loc[i,'objective']))
        ax1.plot(x,model.fls[-1].surface_h,color=color, linewidth=2)
        model.volume_m3_ts().plot(ax=ax3,color=color, linewidth=2)


    for i,model in df['model_t'].iteritems():
        color = cmap(norm(df.loc[i,'objective']))
        ax2.plot(x,model.fls[-1].surface_h,color=color,linewidth=2)

    ax2.plot(x,experiment['y_t'].fls[-1].surface_h,'k:', linewidth=3)
    ax2.plot(x,model.fls[-1].bed_h, 'k', linewidth=3)


    # create volume plot from experiment
    model = experiment['y_t0']
    tasks.run_from_climate_data(gdir, ys=1850, ye=2000,
                                init_model_fls=model.fls,
                                output_filesuffix='experiment')
    rp = gdir.get_filepath('model_run', filesuffix='experiment')
    ex_mod = FileModel(rp)
    ex_mod.volume_m3_ts().plot(ax=ax3, color='k', linestyle=':', linewidth=3)
    ex_mod.run_until(ys)

    ax1.plot(x, ex_mod.fls[-1].surface_h, 'k:', linewidth=3)
    ax1.plot(x, ex_mod.fls[-1].bed_h, 'k', linewidth=3)

    ax1.annotate(r'$t =  '+str(ys)+'$', xy=(0.8,0.9), xycoords='axes fraction',
                 fontsize=15)
    ax2.annotate(r'$t =  2000$', xy=(0.8, 0.9), xycoords='axes fraction',
                 fontsize=15)

    ax1.set_ylabel('Altitude (m)', fontsize=15)
    ax1.set_xlabel('Distance along the main flowline (m)', fontsize=15)
    ax2.set_ylabel('Altitude (m)',fontsize=15)
    ax2.set_xlabel('Distance along the main flowline (m)',fontsize=15)
    ax3.set_ylabel(r'Volume ($m^3$)', fontsize=15)
    ax3.set_xlabel('Time (years)', fontsize=15)

    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cax, kw = mpl.colorbar.make_axes([ax1,ax2,ax3])
    cbar = fig.colorbar(sm,cax=cax,**kw)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('objective', fontsize=15)

    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    ax3.yaxis.offsetText.set_fontsize(15)
    plot_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/past_state_information/plots'
    plt.savefig(os.path.join(plot_dir, 'surface_'+str(ys)+'_'+gdir.rgi_id+'.pdf'), dpi=200)
    plt.savefig(os.path.join(plot_dir, 'surface_' +str(ys)+'_'+ gdir.rgi_id + '.png'), dpi=200)

    #plt.show()
    #plt.close()


def plot_surface(gdir,candidates_df,experiment):
    plot_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/past_state_information/plots'
    #plot_dir = '/home/juliaeis/Dokumente/Präsentationen/AG_Meeting_03_07_18'
    fig, ax1 = plt.subplots(figsize=(25, 15))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    if gdir.name != "":
        ax1.set_title(gdir.rgi_id + ': ' + gdir.name, fontsize=30)
    else:
        ax1.set_title(gdir.rgi_id, fontsize=25)

    x = np.arange(experiment['y_t'].fls[-1].nx) * \
        experiment['y_t'].fls[-1].dx * experiment['y_t'].fls[-1].map_dx

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate(r'$t = t_0 = '+str(1850)+'$', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=30)
    ax2.annotate(r'$t = '+str(2000)+'$', xy=(0.15, 0.85), xycoords='axes fraction',
                 fontsize=25)

    df = candidates_df[candidates_df['objective'] > 100]
    for model in df.model:
        ax1.plot(x, model.fls[-1].surface_h, 'red')
    ax1.plot(x, model.fls[-1].surface_h, 'red', label='not possible')

    for model in df.present_model:
        ax2.plot(x, model.fls[-1].surface_h, 'red')

    df = candidates_df[candidates_df['objective']<=100]
    for model in df.model:
        ax1.plot(x,model.fls[-1].surface_h,'green')
    ax1.plot(x, model.fls[-1].surface_h, 'green', label='possible')
    ax1.plot(x,experiment['y_t0'].fls[-1].surface_h, 'k:',label='experiment', linewidth=2)
    ax1.plot(x,model.fls[-1].bed_h, 'k',label='bed topography')

    for model in df.present_model:
        ax2.plot(x,model.fls[-1].surface_h,'green')
    ax2.plot(x,experiment['y_t'].fls[-1].surface_h, 'k:')
    ax2.plot(x,model.fls[-1].bed_h, 'k')



    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_xlabel('Distance along the main flowline (m)', fontsize=30)
    ax1.set_ylabel('Altitude (m)',fontsize=30)
    ax2.set_xlabel('Distance along the main flowline (m)', fontsize=20)
    ax2.set_ylabel('Altitude (m)', fontsize=20)
    plt.savefig(os.path.join(plot_dir, 'surface.png'),dpi=200)
    plt.show()
    plt.close()

def run_and_store_to_present(i,gdir,ys,ye,flowlines):
    tasks.run_from_climate_data(gdir, ys=1850, ye=2000,
                                init_model_fls=deepcopy(flowlines.iloc[i].fls),
                                output_filesuffix='_past_' + str(i))

def plot_lenght(gdir, candidates_df, experiment):
    plot_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/past_state_information/plots'


    fig,ax = plt.subplots(figsize=(20,10))
    list = range(len(candidates_df))

    pool = Pool()
    present_models = pool.map(partial(run_and_store_to_present, gdir=gdir, ys=1850,
                                      ye=2000,flowlines=candidates_df.model), list)
    pool.close()
    pool.join()

    first1=0
    first2=0
    for i in list:

        if candidates_df.iloc[i].objective <= 100:
            color='green'
            first1=first1+1

        else:
            color='red'
            first2=first2+1
        path = gdir.get_filepath('model_run', filesuffix='_past_' + str(i))
        fmod = FileModel(path)
        if first1 == 1:
            p1=fmod.volume_m3_ts().plot(ax=ax, color=color, label='possible')
            first1=10
        elif first2==1:
            p2 = fmod.volume_m3_ts().plot(ax=ax, color=color, label='not possible')

            first2=10
        else:
            fmod.volume_m3_ts().plot(ax=ax,color=color,label='')


    tasks.run_from_climate_data(gdir, ys=1850, ye=2000,
                                init_model_fls=deepcopy(experiment['y_t0'].fls),
                                output_filesuffix='_past_experiment')
    path = gdir.get_filepath('model_run', filesuffix='_past_experiment')
    fmod = FileModel(path)
    fmod.volume_m3_ts().plot(ax=ax, style='k:',linewidth=3,label='')
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel('time', fontsize=30)
    plt.ylabel(r'volume $(m^3)$',fontsize=30)
    plt.title(gdir.rgi_id,fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(plot_dir, 'lenght_' + str(gdir.rgi_id) + '.png'),
               dpi=200)
    plt.show()
    #plt.close()


def plot_largest_glacier(gdir,candidates_df,ys):
    plot_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/find_initial_state/past_state_information/plots'
    candidates_df = candidates_df[candidates_df['objective']<=100]
    largest = int(candidates_df.model_t0.apply(lambda x: x.volume_m3).idxmax())
    model = candidates_df.loc[largest,'model_t0']
    model.reset_y0(1850)
    graphics.plot_modeloutput_map([gdir],model=candidates_df.loc[largest,'model_t0'])
    plt.savefig(os.path.join(plot_dir, 'largest_'+str(ys)+'_' + str(gdir.rgi_id) + '.png'),
                dpi=200)
    plt.show()
