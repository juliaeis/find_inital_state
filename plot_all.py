from scipy.optimize import minimize
# Scientific packages
import numpy as np
# Constants
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.massbalance import LinearMassBalance
from oggm.core.flowline import FluxBasedModel
from oggm.core.flowline import RectangularBedFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import pickle
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
FlowlineModel = partial(FluxBasedModel, inplace=False)
import math
import xarray as xr
import pandas as pd

def run_model(surface_h):
    nx = 200
    bed_h = np.linspace(3400, 1400, nx)
    # At the begining, there is no glacier so our glacier surface is at the bed altitude

    # Let's set the model grid spacing to 100m (needed later)
    map_dx = 100

    # The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
    widths = np.zeros(nx) + 3.
    # Define our bed
    init_flowline = RectangularBedFlowline(surface_h=rescale(surface_h, nx),
                                         bed_h=bed_h,
                                         widths=widths, map_dx=map_dx)
    # ELA at 3000m a.s.l., gradient 4 mm m-1
    mb_model = LinearMassBalance(3000, grad=4)
#    annual_mb = mb_model.get_mb(surface_h) * SEC_IN_YEAR

    # The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
    model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)
    return model

def rescale(array, mx):
    # interpolate bed_m to resolution of bed_h
    old_indices = np.arange(0, len(array))
    new_length = mx
    new_indices = np.linspace(0, len(array) - 1, new_length)
    spl = UnivariateSpline(old_indices, array, k=1, s=0)
    new_array = spl(new_indices)
    return new_array

orig=run_model(np.linspace(3400, 1400,200))
print(orig.fls[-1].length_m)
orig.reset_y0(0)
orig.run_until(150)
pickle._dump(orig,open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','wb'))
orig.run_until(160)
pickle._dump(orig,open('/home/juliaeis/PycharmProjects/find_inital_state/fls_160.pkl','wb'))

result=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))

result_1=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/result_30pt.txt','rb'))
result_2=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/result_30pt_final.txt','rb'))
result_3=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/result_40pt_final.txt','rb'))
result_4=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/result_15pt.txt','rb'))
result_5=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/result_15pt_noCon5.txt','rb'))
result_6=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/result_40pt.txt','rb'))


model=run_model(result.fls[-1].surface_h)
model1=run_model(result_1)
model2=run_model(result_2)
model3=run_model(result_3)
model4=run_model(result_4)
model5=run_model(result_5)
model6=run_model(result_6)

#print(model6.fls[-1].volume_km3)

x=np.arange(model4.fls[-1].nx) * model4.fls[-1].dx * model4.fls[-1].map_dx



f, axarr = plt.subplots(2, sharex=True)
colors=[]
#plot bed
axarr[0].plot(x,np.linspace(3400,1400,200),'k',label='bed',linewidth=1)
p=axarr[0].plot(x,model1.fls[-1].surface_h,  alpha=0.5,label='initial state 1')
colors.append(p[0].get_color())
axarr[0].plot(model1.length_m,model1.fls[-1].bed_h[int(np.where(x==model1.length_m)[0])],'o',color=p[0].get_color(),alpha=0.5)
p=axarr[0].plot(x,model2.fls[-1].surface_h,  alpha=0.5,label='initial state 2')
colors.append(p[0].get_color())
axarr[0].plot(model2.length_m,model2.fls[-1].bed_h[int(np.where(x==model2.length_m)[0])],'o',color=p[0].get_color(),alpha=0.5)
p=axarr [0].plot(x,model3.fls[-1].surface_h,  alpha=0.5,label='initial state 3')
colors.append(p[0].get_color())
axarr[0].plot(model3.length_m,model3.fls[-1].bed_h[int(np.where(x==model3.length_m)[0])],'o',color=p[0].get_color(),alpha=0.5)
p=axarr[0].plot(x,model4.fls[-1].surface_h,  alpha=0.5,label='initial state 4')
colors.append(p[0].get_color())
axarr[0].plot(model4.length_m,model4.fls[-1].bed_h[int(np.where(x==model4.length_m)[0])],'o',color=p[0].get_color(),alpha=0.5)
p=axarr[0].plot(x,model5.fls[-1].surface_h,  alpha=0.5,label='initial state 5')
colors.append(p[0].get_color())
axarr[0].plot(model5.length_m,model5.fls[-1].bed_h[int(np.where(x==model5.length_m)[0])],'o',color=p[0].get_color(),alpha=0.5)
p=axarr[0].plot(x,model6.fls[-1].surface_h,  alpha=0.5,label='initial state 6')
colors.append(p[0].get_color())
axarr[0].plot(model6.length_m,model6.fls[-1].bed_h[int(np.where(x==model6.length_m)[0])],'o',color=p[0].get_color(),alpha=0.5)
p=axarr[0].plot(x,result.fls[-1].surface_h,  'k',label='idealized glacier')
colors.append(p[0].get_color())

axarr[0].legend(loc='best')
axarr[0].set_ylabel('Altitude (m)')
axarr[0].set_xlabel('Distance along the flowline (m)')
axarr[0].set_title('1850')

model.run_until(300)
model1.run_until(300)
model2.run_until(300)
model3.run_until(300)
model4.run_until(300)
model5.run_until(300)
model6.run_until(300)
'''
model.run_until_and_store(301,run_path='run_model0.txt',diag_path='diag_model0.txt')
model1.run_until_and_store(301,run_path='run_model1.txt',diag_path='diag_model1.txt')
model2.run_until_and_store(301,run_path='run_model2.txt',diag_path='diag_model2.txt')
model3.run_until_and_store(301,run_path='run_model3.txt',diag_path='diag_model3.txt')
model4.run_until_and_store(301,run_path='run_model4.txt',diag_path='diag_model4.txt')
model5.run_until_and_store(301,run_path='run_model5.txt',diag_path='diag_model5.txt')
model6.run_until_and_store(301,run_path='run_model6.txt',diag_path='diag_model6.txt')
'''

axarr[1].plot(x,np.linspace(3400,1400,200),'k',label='bed',linewidth=1)
axarr[1].plot(x,model1.fls[-1].surface_h, label='initial state 1')
axarr[1].plot(x,model2.fls[-1].surface_h, label='initial state 2')
axarr[1].plot(x,model3.fls[-1].surface_h, label='initial state 3')
axarr[1].plot(x,model4.fls[-1].surface_h, label='initial state 4')
plt.plot(x,model5.fls[-1].surface_h, label='initial state 5')
axarr[1].plot(x,model6.fls[-1].surface_h, label='initial state 6')
axarr[1].plot(x,pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb')).fls[-1].surface_h,  'k',label='idealized glacier')
axarr[1].legend(loc='best')
axarr[1].set_ylabel('Altitude (m)')
axarr[1].set_xlabel('Distance along the flowline (m)')
axarr[1].set_title('today- model.run_until')



odf = pd.DataFrame()
vol = pd.DataFrame()

for seed in [ 1,2,3,4,5,6,0]:

    ds = xr.open_dataset('diag_model'+str(seed)+'.txt')

    if seed != 0:
        odf['inital state {}'.format(seed)] = ds.length_m
        vol['inital state {}'.format(seed)] = ds.volume_m3
    else:
        odf['idealized glacier'] = ds.length_m
        vol['idealized glacier'] = ds.volume_m3

odf.index = ds.time+1700
odf.index.name = 'Years'
vol.index = ds.time+1700
vol.index.name = 'Years'

ax = odf.plot(color=colors)
#ax = (odf.rolling(36, center=True).mean()).plot();
ax.set_ylabel('Glacier Length [m]');
ax.set_xlim([1835, 2005])
#plt.title('Glacier Length for some possible initial states')
plt.tight_layout();

plt.figure()
ax = (odf.rolling(36, center=True).mean()).plot(color=colors);
ax.set_ylabel('Glacier Length [m]');
ax.set_xlim([1835, 2005])
#plt.title('glacier ')
plt.tight_layout();


plt.figure()
ax = (vol.rolling(36, center=True).mean()).plot(color=colors);
ax.set_ylabel('Glacier Volume [m^3]');
ax.set_xlim([1835, 2005])
#plt.title('glacier ')
plt.tight_layout();

plt.figure()
plt.plot(x,model1.fls[-1].surface_h-model.fls[-1].surface_h, label='initial state 1')
plt.plot(x,model2.fls[-1].surface_h-model.fls[-1].surface_h, label='initial state 2')
plt.plot(x,model3.fls[-1].surface_h-model.fls[-1].surface_h, label='initial state 3')
plt.plot(x,model4.fls[-1].surface_h-model.fls[-1].surface_h, label='initial state 4')
plt.plot(x,model5.fls[-1].surface_h-model.fls[-1].surface_h, label='initial state 5')
plt.plot(x,model6.fls[-1].surface_h-model.fls[-1].surface_h, label='initial state 6')
plt.legend(loc='best')
plt.title('Differences to idealized glacier')
plt.ylabel('(m)')
plt.xlabel('Distance along the flowline (m)')
plt.show()
