'''
Example of how to use this bayesian optimization package.
'''
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

from bayes_opt import BayesianOptimization

# Lets find the maximum of a simple quadratic function of two variables
# We create the bayes_opt object and pass the function to be maximized
# together with the parameters names and their bounds.
final_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_300.pkl','rb'))
initial_flowline = pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))
def target(ela,time):
    mb_model = LinearMassBalanceModel(ela, grad=4)
    bed_h = np.linspace(3400, 1400, 200)
    # model = FlowlineModel(final_flowline.fls[-1], mb_model=mb_model, y0=0)
    model = FlowlineModel(VerticalWallFlowline(surface_h=bed_h, bed_h=bed_h,
                                               widths=np.zeros(200) + 3.,
                                               map_dx=100), mb_model=mb_model,
                                               y0=0)
    model.run_until(time)
    flowline = model.fls[-1]

    new_mb_model = LinearMassBalanceModel(3000, grad=4)
    new_model = FlowlineModel(flowline, mb_model=new_mb_model, y0=0)
    new_model.run_until(150)
    return -sum(
        abs(final_flowline.fls[-1].surface_h - new_model.fls[-1].surface_h))


bo = BayesianOptimization(target,
                          {'ela': (2500,3500), 'time': (0, 200)})

# One of the things we can do with this object is pass points
# which we want the algorithm to probe. A dictionary with the
# parameters names and a list of values to include in the search
# must be given.
bo.explore({'ela': [3000, 2750], 'time': [0,150]})

# Additionally, if we have any prior knowledge of the behaviour of
# the target function (even if not totally accurate) we can also
# tell that to the optimizer.
# Here we pass a dictionary with 'target' and parameter names as keys and a
# list of corresponding values
#bo.initialize()

# Once we are satisfied with the initialization conditions
# we let the algorithm do its magic by calling the maximize()
# method.
#bo.maximize(init_points=5, n_iter=100, kappa=5)

gp_params = {'kernel': None,
             'alpha': 1e-5}

# Run it again with different acquisition function
bo.maximize(n_iter=10000, acq='ei', **gp_params)

# The output values can be accessed with self.res
print(bo.res['max'])
'''

# If we are not satisfied with the current results we can pickup from
# where we left, maybe pass some more exploration points to the algorithm
# change any parameters we may choose, and the let it run again.
bo.explore({'x': [0.6], 'y': [-0.23]})

# Making changes to the gaussian process can impact the algorithm
# dramatically.
gp_params = {'kernel': None,
             'alpha': 1e-5}

# Run it again with different acquisition function
bo.maximize(n_iter=5, acq='ei', **gp_params)

# Finally, we take a look at the final results.
print(bo.res['max'])
print(bo.res['all'])
'''