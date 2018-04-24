import oggm
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def difference_surface(row):
    return abs(row-solution[1].fls[-1].surface_h)

def difference_surface_1880(row):
    return abs(row-solution[0].fls[-1].surface_h)



def difference_widths(row):
    return abs(row-solution[1].fls[-1].widths)


def difference_widths_1880(row):
    return abs(row-solution[0].fls[-1].widths)


#path= "C://Users//Julia//Desktop//cluster//Ben_idea_parallel"
path = '/home/juliaeis/Schreibtisch/cluster/initializing/run_100'
i=0


for file in os.listdir(path):
    data_1880 = pd.DataFrame()
    data_2000_surface = pd.DataFrame()
    data_2000_widths = pd.DataFrame()
    data_1880_widths = pd.DataFrame()
    res_df = pd.DataFrame(columns=['model_1880','model_2000'])

    if file.endswith('_solution.pkl') :

        solution = pickle.load(open(os.path.join(path,file),'rb'))
        print(file.split('_s')[0], ' ', len(solution[0].fls))

        results = pickle.load(open(os.path.join(path,file.split('_s')[0]+'.pkl'),'rb'))
        for res in results:



            if res[0] != None:
                data_1880=data_1880.append([res[0].fls[-1].surface_h],ignore_index=True)
                data_2000_surface = data_2000_surface.append([res[1].fls[-1].surface_h],ignore_index=True)
                data_2000_widths = data_2000_widths.append([res[1].fls[-1].widths], ignore_index=True)
                data_1880_widths = data_1880_widths.append([res[0].fls[-1].widths],ignore_index=True)

        df = pd.DataFrame(results,columns=['model_1880','model_2000'])
        df=df[~df['model_1880'].isnull()]
        df['length_1880']=[df['model_1880'][i].length_m for i in df.index]
        df['length_2000']=[df['model_2000'][i].length_m for i in df.index]

        df['objective'] = [np.sum(abs(df['model_2000'][i].fls[-1].surface_h - solution[1].fls[-1].surface_h)**2) + \
            np.sum(abs(df['model_2000'][i].fls[-1].widths - solution[1].fls[-1].widths) ** 2) for i in df.index]

        sorted = df.sort_values(by=['objective'], ascending=1)

        lower = data_1880.quantile(q=0.25, axis=0).values
        upper = data_1880.quantile(q=0.75, axis=0).values
        min = data_1880.min(axis=0).values
        max = data_1880.max(axis=0).values

        lower2 = data_2000_surface.quantile(q=0.25, axis=0).values
        upper2 = data_2000_surface.quantile(q=0.75, axis=0).values
        min2 = data_2000_surface.min(axis=0).values
        max2 = data_2000_surface.max(axis=0).values

        lower_w = data_1880_widths.quantile(q=0.25, axis=0).values
        upper_w = data_1880_widths.quantile(q=0.75, axis=0).values
        min_w = data_1880_widths.min(axis=0).values
        max_w = data_1880_widths.max(axis=0).values

        lower2_w = data_2000_widths.quantile(q=0.25, axis=0).values
        upper2_w = data_2000_widths.quantile(q=0.75, axis=0).values
        min2_w = data_2000_widths.min(axis=0).values
        max2_w = data_2000_widths.max(axis=0).values

        # make plot

        x = np.arange(solution[1].fls[-1].nx) * solution[1].fls[-1].dx * solution[1].fls[-1].map_dx

        # **************** plot surface_h media, quartiles and min-max values ***********************************
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
        ax1.set_title(file.split('_s')[0])

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        ax1.annotate('t = 1850', xy=(0.1, 0.95), xycoords='axes fraction', fontsize=13)
        ax2.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',  fontsize=9)

        ax1.plot(x,solution[0].fls[-1].surface_h, 'k:',label='solution')
        ax1.plot(x, solution[0].fls[-1].bed_h, 'k', label='bed topography')
        ax1.plot(x,data_1880.median(axis=0),label='median')
        ax1.plot(x, solution[0].fls[-1].bed_h, 'k')
        ax1.fill_between(x,upper,lower,alpha=0.5,label='25% - 75% quartile')
        ax1.fill_between(x, min, max, alpha=0.2,color='grey',label='range of all possible states')

        ax2.plot(x,data_2000_surface.median(axis=0))
        ax2.fill_between(x, upper2, lower2, alpha=0.5)
        ax2.fill_between(x, min2, max2, alpha=0.2, color='grey')
        ax2.plot(x,solution[1].fls[-1].bed_h, 'k')
        ax2.plot(x,solution[1].fls[-1].surface_h, 'k:')

        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('Distance along the Flowline (m)')
        ax1.set_ylabel('Altitude (m)')

        ax2.set_xlabel('Distance along the Flowline (m)')
        ax2.set_ylabel('Altitude (m)')

        plt.close()

        # **************** plot swidths median, quartiles and min-max values ***********************************
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
        ax1.set_title(file.split('_s')[0])

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        ax1.annotate('t = 1850', xy=(0.1, 0.95), xycoords='axes fraction', fontsize=13)
        ax2.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',  fontsize=9)

        ax1.plot(x,solution[0].fls[-1].widths, 'k:',label='solution')
        ax1.plot(x,data_1880_widths.median(axis=0),label='median')
        ax1.fill_between(x,upper_w,lower_w,alpha=0.5,label='25% - 75% quartile')
        ax1.fill_between(x, min_w, max_w, alpha=0.2,color='grey',label='range of all possible states')

        ax2.plot(x,data_2000_widths.median(axis=0))
        ax2.fill_between(x, upper2_w, lower2_w, alpha=0.5)
        ax2.fill_between(x, min2_w, max2_w, alpha=0.2, color='grey')
        ax2.plot(x,solution[1].fls[-1].widths, 'k:')

        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('Distance along the Flowline (m)')
        ax1.set_ylabel('Widths (m)')

        ax2.set_xlabel('Distance along the Flowline (m)')
        ax2.set_ylabel('Widths (m)')
        plt.close()

        # **************** plot FILTERED surface_h media, quartiles and min-max values ***********************************

        data_1880['objective'] = (data_2000_surface.apply(difference_surface,axis=1)+data_2000_widths.apply(difference_widths,axis=1)).sum(axis=1).sort_values(ascending=True)
        data_1880 = data_1880.sort_values(by='objective')

        filtered = data_1880[data_1880['objective']< data_1880['objective'].median()+data_1880['objective'].mad()].drop(['objective'],axis=1)
        filtered_2000 = data_2000_surface[data_1880['objective'] < data_1880['objective'].median() +data_1880['objective'].mad()]

        filtered_widths = data_1880_widths[data_1880['objective']< data_1880['objective'].median()+data_1880['objective'].mad()]
        filtered_2000_widths = data_2000_widths[data_1880['objective'] < data_1880['objective'].median() +data_1880['objective'].mad()]

        fig, ax3 = plt.subplots(figsize=(20, 10))
        ax4 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
        ax3.set_title(file.split('_s')[0])
        ax3.annotate('t = 1850', xy=(0.1, 0.95), xycoords='axes fraction',
                     fontsize=13)
        ax4.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',
                     fontsize=9)

        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.95, box.height])


        ax3.fill_between(x, filtered.quantile(q=0.25, axis=0).values,
                         filtered.quantile(q=0.75, axis=0).values, alpha=0.5,label='25% - 75% quartile (filtered)')
        ax3.fill_between(x, filtered.min(axis=0).values,
                         filtered.max(axis=0).values, alpha=0.2, color='grey',label='range of all possible states (filtered)')
        ax3.plot(x, solution[0].fls[-1].surface_h, 'k:',label='solution')
        ax3.plot(x, solution[0].fls[-1].bed_h, 'k',label='bed topography')
        ax3.plot(x, filtered.median(axis=0), label='median (filtered)')
        ax3.plot(x, solution[0].fls[-1].bed_h, 'k')
        ax3.plot(x, solution[0].fls[-1].surface_h, 'k:')
        ax4.plot(x,filtered_2000.median(axis=0))
        ax4.fill_between(x, filtered_2000.quantile(q=0.25, axis=0).values,filtered_2000.quantile(q=0.75, axis=0).values, alpha=0.5)
        ax4.fill_between(x, filtered_2000.min(axis=0).values,filtered_2000.max(axis=0).values, alpha=0.2, color='grey')
        ax4.plot(x, solution[1].fls[-1].bed_h, 'k')
        ax4.plot(x, solution[1].fls[-1].surface_h, 'k:')
        ax3.set_title(file.split('_s')[0])


        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_xlabel('Distance along the Flowline (m)')
        ax3.set_ylabel('Altitude (m)')

        ax4.set_xlabel('Distance along the Flowline (m)')
        ax4.set_ylabel('Altitude (m)')
        plt.close()

        # **************** plot FILTERED widths median, quartiles and min-max values ***********************************

        fig, ax3 = plt.subplots(figsize=(20, 10))
        ax4 = fig.add_axes([0.55, 0.66, 0.3, 0.2])
        ax3.set_title(file.split('_s')[0])
        ax3.annotate('t = 1850', xy=(0.1, 0.95), xycoords='axes fraction',
                     fontsize=13)
        ax4.annotate('t = 2000', xy=(0.1, 0.9), xycoords='axes fraction',
                     fontsize=9)

        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        ax3.fill_between(x, filtered_widths.quantile(q=0.25, axis=0).values,
                         filtered_widths.quantile(q=0.75, axis=0).values, alpha=0.5,
                         label='25% - 75% quartile (filtered)')
        ax3.fill_between(x, filtered_widths.min(axis=0).values,
                         filtered_widths.max(axis=0).values, alpha=0.2, color='grey',
                         label='range of all possible states (filtered)')
        ax3.plot(x, solution[0].fls[-1].widths, 'k:', label='solution')
        ax3.plot(x, filtered_widths.median(axis=0), label='median (filtered)')
        ax3.plot(x, solution[0].fls[-1].widths, 'k:')
        ax4.plot(x, filtered_2000_widths.median(axis=0))
        ax4.fill_between(x, filtered_2000_widths.quantile(q=0.25, axis=0).values,
                         filtered_2000_widths.quantile(q=0.75, axis=0).values,
                         alpha=0.5)
        ax4.fill_between(x, filtered_2000_widths.min(axis=0).values,
                         filtered_2000_widths.max(axis=0).values, alpha=0.2,
                         color='grey')
        ax4.plot(x, solution[1].fls[-1].widths, 'k:')
        ax3.set_title(file.split('_s')[0])

        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.set_xlabel('Distance along the Flowline (m)')
        ax3.set_ylabel('Widths(m)')

        ax4.set_xlabel('Distance along the Flowline (m)')
        ax4.set_ylabel('Widths (m)')


        fil = data_1880[data_1880['objective'] < data_1880['objective'].median() + data_1880['objective'].mad()]

        data_1880['error_surface'] = data_1880.drop(['objective'],axis=1).apply(difference_surface_1880,axis=1).sum(axis=1).sort_values(ascending=True)
        data_1880['error_surface_2000'] = data_2000_surface.apply(difference_surface,axis=1).sum(axis=1).sort_values(ascending=True)

        data_1880['error_widths'] = data_1880_widths.apply(difference_widths_1880,axis=1).sum(axis=1).sort_values(ascending=True)
        data_1880['error_widths_2000'] = data_2000_widths.apply(difference_widths,axis=1).sum(axis=1).sort_values(ascending=True)

        data_1880['error_surface_filtered'] = filtered.apply(difference_surface_1880, axis=1).sum(axis=1).sort_values(ascending=True)
        data_1880['error_surface_2000_filtered'] = filtered_2000.apply(difference_surface, axis=1).sum(axis=1).sort_values(ascending=True)

        data_1880['error_widths_filtered'] = filtered_widths.apply(difference_widths_1880, axis=1).sum(axis=1).sort_values(ascending=True)
        data_1880['error_widths_2000_filtered'] = filtered_2000_widths.apply(difference_widths, axis=1).sum(axis=1).sort_values(ascending=True)

        bins=np.linspace(data_1880['error_surface'].min(),data_1880['error_surface'].max(),100)
        bins2 = np.linspace(data_1880['error_surface_2000'].min(),
                           data_1880['error_surface_2000'].max(), 100)
        f,(ax1,ax2) = plt.subplots(2,1)
        #ax1.hist([data_1880['error_surface'],data_1880['error_widths']],bins=50,alpha=0.5, label=['surface', 'widths'])
        ax1.hist([data_1880['error_surface_filtered'].dropna(),data_1880['error_widths_filtered'].dropna()],bins=50,alpha=0.5)
        #ax2.hist([data_1880['error_surface_2000'],data_1880['error_widths_2000']],bins=50,alpha=0.5, label=['surface', 'widths'])
        ax2.hist([data_1880['error_surface_2000_filtered'].dropna(),data_1880['error_widths_2000_filtered'].dropna()],bins=50,alpha=0.5)
        ax1.set_title(file.split('_s')[0])
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Frequency')
        ax1.set_xlabel('difference to solution at t=1880')

        ax2.set_xlabel('difference to solution at t=2000')
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.close()
        #length analysis

        plt.show()