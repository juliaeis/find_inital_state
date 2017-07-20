# The commands below are just importing the necessary modules and functions
# Plot defaults
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (9, 6)  # Default plot size
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
from scipy.optimize import minimize
FlowlineModel = partial(FluxBasedModel, inplace=False)
import copy


final_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))
initial_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))
def objfunc(mb_data):
    mb_model = LinearMassBalanceModel(mb_data[0], grad=4)
    bed_h = np.linspace(3400, 1400, 200)
    model = FlowlineModel(VerticalWallFlowline(surface_h=final_flowline.fls[-1].surface_h, bed_h=bed_h, widths=np.zeros(200) + 3., map_dx=100), mb_model=mb_model,y0=0)
    #model = FlowlineModel(VerticalWallFlowline(surface_h=bed_h, bed_h=bed_h, widths=np.zeros(200) + 3., map_dx=100),mb_model=mb_model,y0=0)
    model.run_until(mb_data[1])
    flowline = model.fls[-1]

    new_mb_model = LinearMassBalanceModel(3000,grad=4)
    new_model = FlowlineModel(flowline, mb_model=new_mb_model, y0=0)
    new_model.run_until(150)
    #print(mb_data,sum(abs(final_flowline.fls[-1].surface_h-new_model.fls[-1].surface_h)))
    return sum(abs(final_flowline.fls[-1].surface_h-new_model.fls[-1].surface_h))+sum(abs(final_flowline.volume_m3))

def con1(mb_data):
    return mb_data -[2000,0]

def con2(mb_data):
    return [5000,3000]-mb_data

x0 = [3000,150]
cons = ({'type': 'ineq', 'fun': con1},
        {'type': 'ineq', 'fun': con2})
res = minimize(objfunc, x0,method='COBYLA',tol=1e-10,constraints=cons,options={'maxiter':5000,'rhobeg' :100})
bed_h = np.linspace(3400, 1400, 200)
model = FlowlineModel(VerticalWallFlowline(surface_h=bed_h, bed_h=bed_h, widths=np.zeros(200) + 3., map_dx=100),mb_model=LinearMassBalanceModel(res.x[0], grad=4),y0=0)
model.run_until(res.x[1])

#inital flowline
plt.figure(0)
plt.plot(initial_flowline.fls[-1].bed_h, color='k', label='Bedrock')
plt.plot(initial_flowline.fls[-1].surface_h, label='Initial flowline')
plt.plot(model.fls[-1].surface_h, label='optimized intial flowline ')
plt.xlabel('Grid points')
plt.ylabel('Altitude (m)')
plt.legend(loc='best');

orig_mb = LinearMassBalanceModel(3000,grad=4)
res_model = FlowlineModel(model.fls[-1],orig_mb,y0=0)
res_model.run_until(150)
plt.figure(1)
plt.plot(initial_flowline.fls[-1].bed_h, color='k', label='Bedrock')
plt.plot(final_flowline.fls[-1].surface_h, label='final flowline')
plt.plot(res_model.fls[-1].surface_h, label='optimized final flowline ')
plt.xlabel('Grid points')
plt.ylabel('Altitude (m)')
plt.legend(loc='best');

plt.figure(2)
plt.plot(initial_flowline.fls[-1].surface_h-model.fls[-1].surface_h, label='difference between original and optimized initial flowline')
#plt.plot(final_flowline.fls[-1].surface_h-res_model.fls[-1].surface_h, label='difference between original and optimized final flowline')
plt.legend(loc='best');
plt.show()