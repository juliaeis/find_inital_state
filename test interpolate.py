from scipy.optimize import minimize
# Scientific packages
import numpy as np
# Constants
from oggm.cfg import SEC_IN_YEAR, A
# OGGM models
from oggm.core.models.massbalance import LinearMassBalanceModel
from oggm.core.models.flowline import FluxBasedModel
from oggm.core.models.flowline import VerticalWallFlowline, \
    TrapezoidalFlowline, ParabolicFlowline
# This is to set a default parameter to a function. Just ignore it for now
from functools import partial
import pickle
import matplotlib.pyplot as plt
import numpy as np
FlowlineModel = partial(FluxBasedModel, inplace=False)
import copy
def rescale(array,mx):
    #interpolate bed_m to resolution of bed_h
    from scipy.interpolate import UnivariateSpline
    old_indices = np.arange(0,len(array))
    new_length = mx
    new_indices = np.linspace(0,len(array)-1,new_length)
    spl = UnivariateSpline(old_indices,array,k=1,s=0)
    new_array = spl(new_indices)
    return new_array

nx = 200
bed_h = np.linspace(3400, 1400, nx)
#original bed
result = ( [3556.82325655 , 3323.4899938 ,  3048.91883072  ,2733.36490086 , 2511.11111111,
  2288.88888889,  2066.66666667  ,1844.44444444  ,1622.22222222,  1400.        ]
)
interp = rescale(result,200)


initial_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))
final_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))

model = FlowlineModel(VerticalWallFlowline(surface_h=interp, bed_h=bed_h, widths=np.zeros(nx) + 3., map_dx=100),mb_model=LinearMassBalanceModel(3000, grad=4),y0=0)
model.run_until(0)
init=copy.deepcopy(model)
model.run_until(150)
#inital flowline
plt.figure(1)
plt.plot(initial_flowline.fls[-1].bed_h, color='k', label='Bedrock')
plt.plot(initial_flowline.fls[-1].surface_h,'--' ,color='teal',label='"initial" glacier - result (normally unknown) -')
plt.plot(final_flowline.fls[-1].surface_h,color='teal', label='glacier after 300 yrs')
plt.plot(init.fls[-1].surface_h, '--',color='coral',label='optimized "intial" glacier')
plt.plot(np.linspace(0,200,len(result)),result, 'o',color='coral')
plt.plot(model.fls[-1].surface_h,color='coral', label='optimized glacier after 300 yrs ')

plt.xlabel('Grid points')
plt.ylabel('Altitude (m)')
plt.legend(loc='best')
#plt.show()
plt.figure(2)
plt.plot(rescale(result,200)-bed_h)
plt.plot(np.zeros(200))
plt.plot(np.zeros(200)+10)

print(np.where(rescale(result,200)-bed_h < 1)[0][0])
plt.show()