
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

x= [ 3624.36798683,  3601.02079106,  3549.1954534 ,  3553.15054129,
        3580.7115955 ,  3518.98118963,  3518.32445852,  3535.66031895,
        3477.22179348,  3449.71982059,  3515.74297767,  3456.23869467,
        3456.63981283,  3439.20357562,  3401.61597897,  3384.65901289,
        3384.80716893,  3368.67813283,  3365.29573581,  3349.47815077,
        3342.49291197,  3327.49601279,  3318.39347331,  3303.50116966,
        3293.96749679,  3278.94110587,  3303.49338342,  3287.62302687,
        3241.20284508,  3227.77208677,  3232.65235287,  3182.79874618,
        3205.62282389,  3156.22184829,  3180.7505826 ,  3126.79125854,
        3110.45465569,  3145.81493974,  3094.7189702 ,  3113.24556726,
        3113.1809388 ,  3124.31107341,  3047.504076  ,  3066.64128635,
        3037.75265989,  3034.75960789,  3022.43149084,  2979.95667377,
        2985.33465101,  2980.61359993,  2906.4263954 ,  2887.43923642,
        2877.38778058,  2870.45227635,  2857.35015715,  2848.79116516,
        2839.5266764 ,  2829.47507783,  2821.00558631,  2809.39488159,
        2801.23330338,  2789.27222617,  2777.7406074 ,  2768.39966508,
        2760.76185382,  2748.76654439,  2739.06309736,  2728.14860196,
        2718.91891187,  2708.1828865 ,  2698.75375058,  2688.76620335,
        2678.76120941,  2668.71133022,  2658.65846351,  2648.54407891,
        2641.52694121,  2629.30727571,  2616.99940663,  2608.5957844 ,
        2596.85354443,  2587.49544452,  2579.36465562,  2566.41558588,
        2555.77915865,  2548.06106608,  2538.03237846,  2527.85051866,
        2517.9080348 ,  2509.60095649,  2497.80590088,  2487.00382719,
        2476.65218344,  2466.91142329,  2457.62944077,  2447.50703432,
        2437.50190474,  2429.37341392,  2417.46325979,  2407.32668705,
        2397.32660465,  2387.22136559,  2374.92262107,  2367.21162852,
        2356.35161967,  2346.34937365,  2337.81175661,  2327.7709874 ,
        2317.76137576,  2306.09930418,  2296.84757377,  2286.00538603,
        2276.74622023,  2265.95615587,  2256.64518817,  2246.54512314,
        2237.31252252,  2226.49373676,  2216.101676  ,  2206.39258365,
        2195.53655887,  2185.51764014,  2176.21994193,  2166.9648198 ,
        2156.11811848,  2146.90515746,  2136.09475947,  2123.61922443,
        2115.88975655,  2106.67723784,  2095.78704523,  2085.73443918,
        2075.75625723,  2064.89135525,  2055.6381171 ,  2045.53093466,
        2034.79850854,  2025.42773388,  2015.34324141,  2003.78416768,
        1995.2741306 ,  1984.51870105,  1975.17047716,  1965.21727622,
        1955.13494045,  1946.60535557,  1935.8077288 ,  1925.79786467,
        1914.86561162,  1904.88285393,  1892.46263104,  1882.45829072,
        1874.57863157,  1864.69962717,  1853.83377401,  1843.84107337,
        1836.89587881,  1824.4776223 ,  1814.426884  ,  1805.15381892,
        1794.26367583,  1785.77827368,  1774.15856452,  1764.1030515 ,
        1754.04649535,  1744.07409328,  1734.02313439,  1724.78962446,
        1712.38357541,  1703.93158577,  1693.82120923,  1682.97847136,
        1673.63712488,  1662.9354568 ,  1655.14895484,  1643.56909768,
        1633.50385289,  1624.24696394,  1613.41734367,  1603.36635657,
        1593.3347037 ,  1583.2511887 ,  1571.58865035,  1563.1827731 ,
        1553.09666597,  1543.86497871,  1532.99864053,  1522.94672839,
        1512.93146978,  1502.04254802,  1492.06866388,  1482.77998357,
        1473.47981581,  1463.419256  ,  1452.59389071,  1442.55719077,
        1432.50603146,  1420.40747812,  1410.32509281,  1400.25020379]

nx = 200
bed_h = np.linspace(3400, 1400, nx)
# At the begining, there is no glacier so our glacier surface is at the bed altitude
surface_h = x
# Let's set the model grid spacing to 100m (needed later)
map_dx = 100

# The units of widths is in "grid points", i.e. 3 grid points = 300 m in our case
widths = np.zeros(nx) + 3.

# Define our bed
init_flowline = VerticalWallFlowline(surface_h=surface_h, bed_h=bed_h, widths=widths, map_dx=map_dx)

# ELA at 3000m a.s.l., gradient 4 mm m-1
mb_model = LinearMassBalanceModel(3000, grad=4)
annual_mb = mb_model.get_mb(bed_h) * SEC_IN_YEAR


# The model requires the initial glacier bed, a mass-balance model, and an initial time (the year y0)
model = FlowlineModel(init_flowline, mb_model=mb_model, y0=150)

model.run_until(300)
#load original initial glacier
orig=pickle.load(open('/home/juliaeis/PycharmProjects/find_inital_state/fls_150.pkl','rb'))
orig_model = FlowlineModel(orig.fls[-1],mb_model=mb_model,y0=150)
orig_model.run_until(300)
plt.figure(1)
# Plot the initial conditions first:
plt.plot(init_flowline.bed_h, color='k', label='Bedrock')
plt.plot(init_flowline.surface_h,color='r', label='Initial glacier')
# The get the modelled flowline (model.fls[-1]) and plot it's new surface
plt.plot(orig.fls[-1].surface_h, color='b', label='original initial glacier ')
plt.plot(model.fls[-1].surface_h, color='r', label='after 300 years')
plt.plot(orig_model.fls[-1].surface_h, color='b', label='original after 300 years')
plt.xlabel('Grid points')
plt.ylabel('Altitude (m)')
plt.legend(loc='best');

plt.figure(2)
plt.plot(orig_model.fls[-1].surface_h-model.fls[-1].surface_h, color='b', label='difference')
plt.show()
