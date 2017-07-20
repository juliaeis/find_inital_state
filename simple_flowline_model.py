# The commands below are just importing the necessary modules and functions
# Plot defaults
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (9, 6)  # Default plot size
# Scientific packages
import numpy as np
# Constants
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.models.massbalance import LinearMassBalanceModel
from oggm.core.models.flowline import FluxBasedModel
from oggm.core.models.flowline import VerticalWallFlowline, TrapezoidalFlowline, ParabolicFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import pickle
FlowlineModel = partial(FluxBasedModel, inplace=False)


#glacier  bed
# This is the bed rock, linearily decreasing from 3000m altitude to 1000m, in 200 steps
nx = 200
bed_h = np.linspace(3400, 1400, nx)
# At the begining, there is no glacier so our glacier surface is at the bed altitude
surface_h = bed_h
# Let's set the model grid spacing to 100m (needed later)
map_dx = 100

# The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
widths = np.zeros(nx) + 3.

# Define our bed
init_flowline = VerticalWallFlowline(surface_h=surface_h, bed_h=bed_h, widths=widths, map_dx=map_dx)

# ELA at 3000m a.s.l., gradient 4 mm m-1
mb_model = LinearMassBalanceModel(3000, grad=4)
annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR


# The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
model = FlowlineModel(init_flowline, mb_model=mb_model, y0=0)
model.run_until(150)
print(model.length_m, model.area_km2, model.volume_km3)
d= model.fls[-1].to_dataset()
pickle.dump(model,open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','wb'))

model.run_until(300)
print(model.length_m, model.area_km2, model.volume_km3)
pickle.dump(model,open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','wb'))