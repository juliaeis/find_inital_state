from oggm import cfg,workflow,tasks
from oggm.utils import get_demo_file
import pickle
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import copy
import salem

def diff(ar1,ar2):
    return abs(ar1-ar2)


def plot_surface (data_1880, data_2000,solution,rgi_id,fls_num, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if 'objective' in data_2000.columns:
        data_2000 = data_2000.drop('objective',axis=1)

    x = np.arange(solution[1].fls[fls_num].nx) * solution[1].fls[fls_num].dx * \
        solution[1].fls[-1].map_dx

    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    ax1.set_title(rgi_id+', flowline '+ str(fls_num))
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate('t = 1880', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=18)
    ax2.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',
                 fontsize=13)

    ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:', label='solution'
             )
    ax1.plot(x, solution[0].fls[fls_num].bed_h, 'k', label='bed')
    ax1.plot(x, data_1880.median(axis=0), label='median')
    ax1.plot(x, solution[0].fls[fls_num].surface_h, 'k:')
    ax1.plot(x, solution[0].fls[fls_num].bed_h, 'k')

    ax1.fill_between(x, data_1880.quantile(q=0.75, axis=0).values,
                     data_1880.quantile(q=0.25, axis=0).values,
                     alpha=0.5, label='25% - 75% quartile')
    ax1.fill_between(x, data_1880.min(axis=0).values,
                     data_1880.max(axis=0).values, alpha=0.2, color='grey',
                     label='range of all possible states')

    ax2.plot(x, data_2000.median(axis=0))
    ax2.fill_between(x, data_2000.quantile(q=0.75, axis=0).values,
                     data_2000.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, data_2000.min(axis=0).values,
                     data_2000.max(axis=0).values, alpha=0.2,color='grey')

    ax2.plot(x, solution[1].fls[fls_num].bed_h, 'k')
    ax2.plot(x, solution[1].fls[fls_num].surface_h, 'k:')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel('Distance along the Flowline')
    ax1.set_ylabel('Altitude (m)')

    ax2.set_xlabel('Distance along the Flowline (m)')
    ax2.set_ylabel('Altitude (m)')

    plt.savefig(os.path.join(plot_dir,str(rgi_id)+'_fls'+str(fls_num)+'_surface.png'))
    #plt.close()
    return


def plot_widths(data_1880, data_2000,solution,rgi_id,fls_num, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    x = np.arange(solution[1].fls[fls_num].nx) * solution[1].fls[fls_num].dx * \
        solution[1].fls[fls_num].map_dx

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
    ax1.set_title(rgi_id + ', flowline ' + str(fls_num))
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax1.annotate('t = 1880', xy=(0.1, 0.95), xycoords='axes fraction',
                 fontsize=13)
    ax2.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',
                 fontsize=9)

    ax1.plot(x, solution[0].fls[fls_num].widths, 'k:', label='solution')
    ax1.plot(x, data_1880.median(axis=0), label='median')
    ax1.plot(x, solution[0].fls[fls_num].widths, 'k:')
    ax1.fill_between(x, data_1880.quantile(q=0.75, axis=0).values,
                     data_1880.quantile(q=0.25, axis=0).values,
                     alpha=0.5, label='25% - 75% quartile')
    ax1.fill_between(x, data_1880.min(axis=0).values,
                     data_1880.max(axis=0).values, alpha=0.2, color='grey',
                     label='range of all possible states')

    ax2.plot(x, data_2000.median(axis=0))
    ax2.fill_between(x, data_2000.quantile(q=0.75, axis=0).values,
                     data_2000.quantile(q=0.25, axis=0).values,
                     alpha=0.5)
    ax2.fill_between(x, data_2000.min(axis=0).values,
                     data_2000.max(axis=0).values, alpha=0.2,color='grey')
    ax2.plot(x, solution[1].fls[fls_num].widths, 'k:')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel('Distance along the Flowline (m)')
    ax1.set_ylabel('Widths (m)')

    ax2.set_xlabel('Distance along the Flowline (m)')
    ax2.set_ylabel('Widths (m)')

    plt.savefig(os.path.join(plot_dir,str(rgi_id)+'_fls'+str(fls_num)+'_widths.png'))
    plt.close()
    return


def plot_difference_histogramm(diff_s_1880,diff_s_2000,diff_w_1880,diff_w_2000,
                               plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # plot histogramm difference

    f, ax = plt.subplots(2, 2,figsize=(20,10))
    main_fls_num = sorted(diff_w_1880.keys())[-1]
    ax[0][0].hist([diff_s_1880[i] for i in diff_s_1880.keys()],
                  label=['fls_' + str(j) for j in diff_s_1880.keys()], bins=20)
    ax[1][0].hist([diff_s_2000[i] for i in diff_s_2000.keys()],
                  label=['fls_' + str(j) for j in diff_s_2000.keys()], bins=20)
    ax[0][1].hist([diff_w_1880[i] for i in diff_w_1880.keys()],
                  label=['fls_' + str(j) for j in diff_w_1880.keys()], bins=20)
    ax[1][1].hist([diff_w_2000[i] for i in diff_w_2000.keys()],
                  label=['fls_' + str(j) for j in diff_w_2000.keys()], bins=20)

    plt.suptitle(rgi_id)
    ax[0][0].set_ylabel('Frequency')
    ax[0][0].set_xlabel('Surface Error at t=1880')

    ax[1][0].set_ylabel('Frequency')
    ax[1][0].set_xlabel('Surface Error at t=2000')

    ax[0][1].set_ylabel('Frequency')
    ax[0][1].set_xlabel('Widths Error at t=1880')

    ax[1][1].set_ylabel('Frequency')
    ax[1][1].set_xlabel('Widths Error at t=2000')

    ax[0][0].legend(loc='best')
    ax[1][0].legend(loc='best')
    ax[0][1].legend(loc='best')
    ax[1][1].legend(loc='best')

    plt.savefig(os.path.join(plot_dir, str(rgi_id) + '_difference_histogramm.png'))
    # plt.close()


def plot_difference_boxplot(diff_s_1880, diff_s_2000, diff_w_1880, diff_w_2000, plot_dir):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # plot histogramm difference

    f, ax = plt.subplots(2, 2, figsize=(20, 10))
    main_fls_num = sorted(diff_w_1880.keys())[-1]

    ax[0][0].boxplot([diff_s_1880[i] for i in diff_s_1880.keys()], labels=['fls_'+str(i) for i in diff_s_1880.keys()])
    ax[1][0].boxplot([diff_s_2000[i] for i in diff_s_2000.keys()], labels=['fls_'+str(i) for i in diff_s_1880.keys()])
    ax[0][1].boxplot([diff_w_1880[i] for i in diff_w_1880.keys()], labels=['fls_'+str(i) for i in diff_s_1880.keys()])
    ax[1][1].boxplot([diff_w_2000[i] for i in diff_w_2000.keys()], labels=['fls_'+str(i) for i in diff_s_1880.keys()])

    plt.suptitle(rgi_id)

    ax[0][0].set_ylabel('Surface Error at t=1880')
    ax[1][0].set_ylabel('Surface Error at t=2000')
    ax[0][1].set_ylabel('Widths Error at t=1880')
    ax[1][1].set_ylabel('Widths Error at t=2000')


    plt.savefig(
        os.path.join(plot_dir, str(rgi_id) + '_difference_histogramm.png'))
    plt.close()


def plot_difference_length(results,solution, rgi_id, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    length = pd.DataFrame() 
    for res in results:
        if res[0] != None:
            length = length.append(pd.Series(
                {'1880': solution[0].length_m - res[0].length_m,
                 '2000': solution[1].length_m - res[1].length_m}),
                                   ignore_index=True)
    length.plot.box()
    plt.show()


if __name__ == '__main__':

    path = '/home/juliaeis/Schreibtisch/cluster/initializing/run_100'
    plot_dir = os.path.join(path,'analysis_plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    analysis = pd.DataFrame()
    analysis2 = pd.DataFrame()
    for file in os.listdir(path):


        s_1880 = {}
        s_2000 = {}
        w_1880 = {}
        w_2000 = {}

        diff_s_1880 = {}
        diff_s_2000 = {}
        diff_w_1880 = {}
        diff_w_2000 = {}

        '''
        if file.endswith('_solution.pkl') :
            rgi_id = file.split('_s')[0]
            solution_file = os.path.join(path,file)
            result_file = os.path.join(path,rgi_id+'.pkl')

            solution = pickle.load(open(solution_file,'rb'))
            results = pickle.load(open(result_file,'rb'))
            surface_1880 = pd.DataFrame()
            for res in results:
                if res[0] !=None:
                    surface_1880 = surface_1880.append([res[0].fls[-1].surface_h], ignore_index=True)
            median = np.array(surface_1880.median(axis=0)-solution[0].fls[-1].bed_h)
            pixel_diff = np.where(median>0)[0][-1]-np.where(solution[0].fls[-1].surface_h-solution[0].fls[-1].bed_h >0)[0][-1]
            len_diff = pixel_diff*solution[1].fls[-1].dx *solution[1].fls[-1].map_dx
            analysis2 = analysis2.append({'RGIId': rgi_id, 'len_diff': int(len_diff)}, ignore_index=True)

            for i in range(len(solution[0].fls)):
                surface_1880 = pd.DataFrame()
                surface_2000 = pd.DataFrame()
                widths_1880 = pd.DataFrame()
                widths_2000 = pd.DataFrame()
                length_1880 = pd.DataFrame()
                for res in results:
                    if res[0] != None:
                        # surface
                        surface_1880 = surface_1880.append([res[0].fls[i].surface_h],ignore_index=True)
                        surface_2000 = surface_2000.append([res[1].fls[i].surface_h],ignore_index=True)
                        # widths
                        widths_2000 = widths_2000.append([res[1].fls[i].widths], ignore_index=True)
                        widths_1880 = widths_1880.append([res[0].fls[i].widths],ignore_index=True)
                        length_1880 = length_1880.append([res[0].length_m],ignore_index=True)

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

            for i in range(len(s_1880)):

                s_1880[i] = s_1880[i][filtered]
                s_2000[i] = s_2000[i][filtered]
                w_1880[i] = w_1880[i][filtered]
                w_2000[i] = w_2000[i][filtered]


                plot_surface(s_1880[i], s_2000[i], solution,rgi_id,i, os.path.join(plot_dir,'surface'))
                plot_widths(w_1880[i], w_2000[i], solution, rgi_id,i, os.path.join(plot_dir,'widths'))

                # calculate difference for all flowlines

                diff_s_1880[i] = s_1880[i].apply(diff, axis=1, args=[solution[0].fls[i].surface_h]).sum(axis=1)
                diff_s_2000[i] = s_2000[i].apply(diff, axis=1, args=[solution[1].fls[i].surface_h]).sum(axis=1)
                diff_w_1880[i] = w_1880[i].apply(diff, axis=1, args=[solution[0].fls[i].widths]).sum(axis=1)
                diff_w_2000[i] = w_2000[i].apply(diff, axis=1,args=[solution[1].fls[i].widths]).sum(axis=1)



            plot_difference_histogramm(diff_s_1880, diff_s_2000, diff_w_1880, diff_w_2000, os.path.join(plot_dir,'difference'))
            plot_difference_boxplot(diff_s_1880, diff_s_2000, diff_w_1880, diff_w_2000, os.path.join(plot_dir,'boxplot'))
            plot_difference_length(results,solution, rgi_id, plot_dir)

            plt.show()

            analysis = analysis.append({'RGIId':rgi_id,'range':int(length_1880.max()-length_1880.min())},ignore_index=True)
    analysis = analysis.set_index('RGIId')
    pickle.dump(analysis,open(os.path.join(path,'analysis.txt'),'wb'))
    '''
    analysis = pickle.load(open(os.path.join(path,'analysis.txt'),'rb'))
    #analysis2 = analysis2.set_index('RGIId')
    #analysis = pd.concat([analysis,analysis2],axis=1)
    #pickle.dump(analysis, open(os.path.join(path, 'analysis.txt'), 'wb'))

    rgi_file = get_demo_file('rgi_oetztal.shp')
    # rgi = get_demo_file('HEF_MajDivide.shp')
    rgi = salem.read_shapefile(rgi_file)

    rgi = rgi.set_index('RGIId')
    analysis = pd.concat([analysis,rgi['Slope']],axis=1)
    pd.scatter_matrix(analysis, figsize=(6, 6), diagonal='hist')
    plt.show()