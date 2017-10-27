import os
import geopandas as gpd
import oggm
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from oggm import cfg, tasks
from oggm.utils import get_demo_file
from oggm.core.preprocessing.climate import (mb_yearly_climate_on_glacier,
                                             t_star_from_refmb,
                                             local_mustar_apparent_mb)
from oggm.core.models.massbalance import (PastMassBalanceModel,
                                          ConstantMassBalanceModel)
import matplotlib.pyplot as plt
from oggm.core.models.flowline import FluxBasedModel


import oggm
from oggm import cfg
from oggm.utils import get_demo_file
cfg.initialize()
srtm_f = get_demo_file('srtm_oetztal.tif')
rgi_f = get_demo_file('rgi_oetztal.shp')
print(srtm_f)

import salem  # https://github.com/fmaussion/salem
rgi_shp = salem.read_shapefile(rgi_f).set_index('RGIId')

# Plot defaults

# Packages
import os
import numpy as np
import xarray as xr
import shapely.geometry as shpg
plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size


from oggm import cfg
from oggm import workflow
cfg.initialize()  # read the default parameter file
cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
cfg.PARAMS['border'] = 80
cfg.PARAMS['prcp_scaling_factor']
import pickle

# Read in the RGI file
import geopandas as gpd
rgi_file = get_demo_file('rgi_oetztal.shp')
rgidf = gpd.GeoDataFrame.from_file(rgi_file)
# Initialise directories
# reset=True will ask for confirmation if the directories are already present:
# this is very useful if you don't want to loose hours of computations because of a command gone wrong

gdirs = workflow.init_glacier_regions(rgidf)
from oggm import graphics
gdir = gdirs[13]

#workflow.execute_entity_task(tasks.glacier_masks, gdirs)
gdir_hef = [gd for gd in gdirs if (gd.rgi_id == 'RGI50-11.00897')][0]
'''
list_talks = [
         tasks.compute_centerlines,
         tasks.compute_downstream_lines,
         tasks.catchment_area,
         tasks.initialize_flowlines,
         tasks.catchment_width_geom,
         tasks.catchment_width_correction,
         tasks.compute_downstream_bedshape
         ]

for task in list_talks:
    workflow.execute_entity_task(task, gdirs)

workflow.climate_tasks(gdirs)
workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs)
from oggm.core.preprocessing.inversion import mass_conservation_inversion

# Select HEF out of all glaciers

glen_a = cfg.A
vol_m3, area_m3 = mass_conservation_inversion(gdir_hef, glen_a=glen_a)
print('With A={}, the mean thickness of HEF is {:.1f} m'.format(glen_a, vol_m3/area_m3))
optim_resuls = tasks.optimize_inversion_params(gdirs)

workflow.execute_entity_task(tasks.volume_inversion, gdirs)
workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)'''
tasks.init_present_time_glacier(gdir_hef)
fls = gdir_hef.read_pickle('model_flowlines')
model = FluxBasedModel(fls)
surface_before=model.fls[-1].surface_h
from oggm.core.models.massbalance import ConstantMassBalanceModel
from oggm.core.models.massbalance import PastMassBalanceModel
today_model = ConstantMassBalanceModel(gdir_hef, y0=1985)

commit_model = FluxBasedModel(fls, mb_model=today_model, glen_a=cfg.A)
pickle.dump(gdir_hef,open('gdir_hef.pkl','wb'))
plt.figure(0)
#graphics.plot_modeloutput_section(gdir_hef, model=commit_model)
print(commit_model.length_m)

commit_model.run_until(100)

x=np.arange(model.fls[-1].nx) * model.fls[-1].dx * model.fls[-1].map_dx
plt.figure(1)

plt.plot(x,model.fls[-1].bed_h,'k')
plt.plot(x,surface_before,color='teal', label='y0')
plt.plot(x,commit_model.fls[-1].surface_h,color='tomato',label='y50')
plt.legend(loc='best')
plt.xlabel('Distance along the flowline (m)')
plt.ylabel('Altitude (m)')
plt.show()


