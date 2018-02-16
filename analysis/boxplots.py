import oggm
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path= "C://Users//Julia//Desktop//cluster//Ben_idea_parallel"
i=0
for file in os.listdir(path):
    res_df = pd.DataFrame(columns=['model_1880','model_2000'])

    if file.endswith('_solution.pkl'):

        solution = pickle.load(open(os.path.join(path,file),'rb'))


        results = pickle.load(open(os.path.join(path,file.split('_')[0]+'.pkl'),'rb'))

        df = pd.DataFrame(results,columns=['model_1880','model_2000'])
        df=df[~df['model_1880'].isnull()]
        df['length_1880']=[df['model_1880'][i].length_m for i in df.index]
        df['length_2000']=[df['model_2000'][i].length_m for i in df.index]

        df['objective'] = [np.sum(abs(df['model_2000'][i].fls[-1].surface_h - solution[1].fls[-1].surface_h)**2) + \
            np.sum(abs(df['model_2000'][i].fls[-1].widths - solution[1].fls[-1].widths) ** 2) + \
            abs(df['model_2000'][i].length_m - solution[1].length_m)**2 for i in df.index]

        print(df['length_2000'])

        plt.figure()

        for i in  df[df['objective']<100].index:
            plt.plot(df.loc[i,'model_2000'].fls[-1].surface_h,'r')
        for i in df[df['objective']>100].index:
            plt.plot(df.loc[i,'model_2000'].fls[-1].surface_h,'b', alpha=0.5)
        plt.plot(solution[0].fls[-1].bed_h,'k')
        #plt.plot(solution[0].fls[-1].widths,'k:')

        #df['objective'].plot.hist()
        #plt.axhline(solution[0].length_m, color='k')
        plt.show()
        i=i+1

