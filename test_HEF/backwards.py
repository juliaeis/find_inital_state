from oggm import cfg, workflow, tasks
from oggm.utils import get_demo_file
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.massbalance import LinearMassBalance, PastMassBalance, RandomMassBalance,ConstantMassBalance
from oggm.core.flowline import FluxBasedModel,RectangularBedFlowline
from functools import partial
FlowlineModel = partial(FluxBasedModel, inplace=False)
from bayes_opt import BayesianOptimization

import os
import salem
import copy
import numpy as np
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from scipy.ndimage.filters import median_filter,gaussian_filter

if __name__ == '__main__':
    start_time = time.time()
    cfg.initialize()
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
    cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
    cfg.PATHS['working_dir'] = '/home/juliaeis/PycharmProjects/find_inital_state/test_HEF'
    #cfg.PATHS['working_dir'] = os.environ.get("S_WORKDIR")
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['prcp_scaling_factor']
    cfg.PARAMS['run_mb_calibration'] = True
    cfg.PARAMS['optimize_inversion_params'] = True

    plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size
    '''
    rgi = get_demo_file('rgi_oetztal.shp')
    gdirs = workflow.init_glacier_regions(salem.read_shapefile(rgi))
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)
    gdir=gdirs[0]

    fls = gdir.read_pickle('model_flowlines')
    past_climate = PastMassBalance(gdir)
    commit_model = FluxBasedModel(fls, mb_model=past_climate,
                                  glen_a=cfg.A, y0=2000, time_stepping='ambitious')
    '''
    nx = 200
    bed_h = np.linspace(3400, 1400, nx)
    # At the begining, there is no glacier so our glacier surface is at the bed altitude
    surface_h = bed_h
    # Let's set the model grid spacing to 100m (needed later)
    map_dx = 100

    # The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
    widths = np.zeros(nx) + 3.

    # Define our bed
    init_flowline = RectangularBedFlowline(surface_h=surface_h, bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)

    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalance(3000, grad=4)
    commit_model = FlowlineModel(init_flowline, mb_model=mb_model, y0=0)
    commit_model.run_until(300)

    constant_climate=LinearMassBalance(3000, grad=3)
    commit_model = FlowlineModel(commit_model.fls, mb_model=constant_climate, y0=200)
    y_start = copy.deepcopy(commit_model)

    x = np.arange(y_start.fls[-1].nx) * y_start.fls[-1].dx * y_start.fls[-1].map_dx
    commit_model.run_until(297)
    y_270 = copy.deepcopy(commit_model)
    commit_model.run_until(300)
    y_300 =  copy.deepcopy(commit_model)
    commit_model.reset_y0(300)
    commit_model.run_until_back(297)
    commit_model.reset_y0(297)
    y_270_back = copy.deepcopy(commit_model)
    '''
    signs = np.sign(np.gradient(y_270_back.fls[-1].section))

    #if len(np.where(np.abs(np.diff(signs))>1)[0]>0):
        start_filter = np.where(np.abs(np.diff(signs))>1)[0][0]
        end_filter = np.where(np.abs(np.diff(signs))>1)[0][-1]
        a = y_270_back.fls[-1].section[:int(start_filter-10)]
        b = median_filter(y_270_back.fls[-1].section[int(start_filter - 10):int(end_filter+10)],size=11,mode='nearest')
        c = y_270_back.fls[-1].section[int(end_filter+10):]
        filt_section=(np.concatenate((a,b,c)))
    '''

    #commit_model.run_until(10)

    #print(commit_model.yr)

    plt.figure()

    #ax1 = plt.subplot(211)
    #ax1.set_title(gdir.rgi_id)
    #plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot( y_270_back.fls[-1].surface_h,
             label='t=270_backwards')
    #y_270_back.fls[-1].section = filt_section

    #plt.plot( gaussian_filter(y_270.fls[-1].surface_h,sigma=3, mode='nearest'))
    plt.plot( y_270.fls[-1].surface_h, label='t=270 solution')
    plt.plot(y_300.fls[-1].surface_h, label='t=300 solution')
    plt.plot(commit_model.fls[-1].bed_h,'k')

    plt.legend(loc='best')
    '''
    ax2 = plt.subplot(212, sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.plot(x, y_250.fls[-1].surface_h, 'k:', label='solution')
    ax2.plot(x, y_250.fls[-1].bed_h, 'k', label='bed')
    '''
    plt.show()