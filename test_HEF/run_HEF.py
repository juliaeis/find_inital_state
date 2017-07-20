import os
import geopandas as gpd
import oggm
import salem
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from oggm import cfg, tasks,workflow,graphics
from oggm.utils import get_demo_file
from oggm.core.preprocessing.climate import (mb_yearly_climate_on_glacier,
                                             t_star_from_refmb,
                                             local_mustar_apparent_mb)
from oggm.core.models.massbalance import (PastMassBalanceModel,
                                          ConstantMassBalanceModel)
from oggm.core.models.flowline import FluxBasedModel
from oggm.core.preprocessing.inversion import mass_conservation_inversion

# Plot defaults
#%matplotlib inline
import matplotlib.pyplot as plt
# Packages
import os
import numpy as np
import xarray as xr
import shapely.geometry as shpg

plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size
if __name__ == '__main__':

    cfg.initialize()
    srtm_f = get_demo_file('srtm_oetztal.tif')
    rgi_f = get_demo_file('rgi_oetztal.shp')
    rgi_shp = salem.read_shapefile(rgi_f).set_index('RGIId')
    #rgi_shp.plot()


    cfg.PATHS['dem_file']=srtm_f
    cfg.PATHS['climate_file']=get_demo_file('HISTALP_oetztal.nc')
    base_dir = os.path.join('/home/juliaeis/Dokumente/OGGM/work_dir','find_initial_state_HEF')
    cfg.PARAMS['border']=80
    cfg.PATHS['working_dir']=base_dir
    rgidf = gpd.GeoDataFrame.from_file(rgi_f)
    gdirs =oggm.workflow.init_glacier_regions(rgidf, reset=False)
    '''
    list_talks = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.compute_downstream_lines,
        tasks.catchment_area,
        tasks.initialize_flowlines,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.compute_downstream_bedshape
        ]
    for task in list_talks:
        workflow.execute_entity_task(task,gdirs)

    # climate
    #workflow.climate_tasks(gdirs)
    #workflow.execute_entity_task(tasks.prepare_for_inversion,gdirs)
    '''
    #only HEF
    gdir_hef = [gd for gd in gdirs if (gd.rgi_id ==  'RGI50-11.00897')][0]
    '''
    glen_a = cfg.A
    vol_m3,area_m3 = mass_conservation_inversion(gdir_hef,glen_a=glen_a)
    print('With A={}, the mean thickness of HEF is {:.1f} m'.format(glen_a, vol_m3/area_m3))
    graphics.plot_inversion(gdir_hef,add_scalebar=False)

    optim_results = tasks.optimize_inversion_params(gdirs)

    workflow.execute_entity_task(tasks.volume_inversion, gdirs)
    workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
    '''
    tasks.init_present_time_glacier(gdir_hef)
    fls = gdir_hef.read_pickle('model_flowlines')
    model = FluxBasedModel(fls)
    #graphics.plot_modeloutput_map(gdir_hef,model=model,add_scalebar=False)
    graphics.plot_centerlines(gdir_hef,add_scalebar=False)
    #plt.show()
    for i in range(len(model.fls)):
        plt.figure(i)
        plt.plot(model.fls[i].bed_h)
        plt.plot(model.fls[i].surface_h)
    plt.show()