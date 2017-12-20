import os
import geopandas as gpd
import oggm
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from oggm import cfg, tasks
from oggm.utils import get_demo_file
from oggm.core.climate import (mb_yearly_climate_on_glacier, t_star_from_refmb)
from oggm.core.massbalance import (PastMassBalance, ConstantMassBalance)
from oggm.core.flowline import FluxBasedModel
import oggm
from oggm import cfg, workflow,graphics
from oggm.utils import get_demo_file
import salem
import shapely.geometry as shpg
import pickle

plt.rcParams['figure.figsize'] = (8, 8)  # Default plot size
cfg.initialize()  # read the default parameter file
cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')
cfg.PATHS['climate_file'] = get_demo_file('HISTALP_oetztal.nc')
cfg.PATHS['working_dir']='/home/juliaeis/PycharmProjects/find_inital_state/test_HEF'
cfg.PARAMS['border'] = 80
cfg.PARAMS['prcp_scaling_factor']
cfg.PARAMS['run_mb_calibration'] = True
cfg.PARAMS['optimize_inversion_params']=True


srtm_f = get_demo_file('srtm_oetztal.tif')
rgi_f = get_demo_file('rgi_oetztal.shp')

rgi_shp = salem.read_shapefile(rgi_f)

gdirs = workflow.init_glacier_regions(rgi_shp)
#gdir = gdirs[13]

workflow.execute_entity_task(tasks.glacier_masks, gdirs)
gdir_hef = [gd for gd in gdirs if (gd.rgi_id == 'RGI50-11.00897')][0]

list_talks = [
         tasks.compute_centerlines,
         tasks.compute_downstream_line,
         tasks.compute_downstream_bedshape,
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
from oggm.core.inversion import mass_conservation_inversion

# Select HEF out of all glaciers

glen_a = cfg.A
vol_m3, area_m3 = mass_conservation_inversion(gdir_hef, glen_a=glen_a)
print('With A={}, the mean thickness of HEF is {:.1f} m'.format(glen_a, vol_m3/area_m3))
optim_resuls = tasks.optimize_inversion_params(gdirs)

workflow.execute_entity_task(tasks.volume_inversion, gdirs)
workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
tasks.init_present_time_glacier(gdir_hef)
fls = gdir_hef.read_pickle('model_flowlines')

model = FluxBasedModel(fls)
surface_before=model.fls[-1].surface_h

today_model = ConstantMassBalance(gdir_hef, y0=1985)

commit_model = FluxBasedModel(fls, mb_model=today_model, glen_a=cfg.A)
pickle.dump(gdir_hef,open('gdir_hef.pkl','wb'))
pickle.dump(commit_model.fls,open('hef_y0.pkl','wb'))
plt.figure(0)
#graphics.plot_modeloutput_section(gdir_hef, model=commit_model)
print(commit_model.length_m)

commit_model.run_until(100)
pickle.dump(commit_model.fls,open('hef_y1.pkl','wb'))
x=np.arange(model.fls[-1].nx) * model.fls[-1].dx * model.fls[-1].map_dx
plt.figure(1)
plt.plot(x,model.fls[-1].bed_h,'k')
plt.plot(x,surface_before,color='teal', label='y0')
plt.plot(x,commit_model.fls[-1].surface_h,color='tomato',label='y1')
plt.legend(loc='best')
plt.xlabel('Distance along the flowline (m)')
plt.ylabel('Altitude (m)')
plt.show()


