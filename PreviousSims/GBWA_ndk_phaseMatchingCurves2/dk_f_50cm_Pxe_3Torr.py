import functools
import pandas as pd
import numpy as np
import tqdm
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as scint
import scipy.interpolate as scinterp
xr.set_options(keep_attrs=True)
import cmath
import sys
from Functions import *



pulse_params = {'b' : f_to_b(omega0_355,50e-2),            # [m] confocal parameter
            'energy' : 17e-3,       # [J] single pulse energy original
            'duration' : 7e-9,       # [s] single pulse length
            'PXe':3,
            'PAr':25*11.5}    
harm_params = {'alpha' : 8.20839154e-48, #value from fit 'alpha' : 8.20839154e-48,
            'chi3' : 1.5e-35}   #value from fit: 1.5e-35

zstart = -0.2
zstop =0.2
zsamples = 1000
zrange = (zstart, zstop)
z = np.array(np.linspace(zstart, zstop, zsamples,dtype = np.longdouble))

rstop = 10*omega0_355
rsamples = 10000
r = np.linspace(0,rstop,rsamples,dtype = np.longdouble)


sol_params_WA = {'func': curly_GBWA,
              'initial_vals': (nonzero, nonzero),
              'zstart': zstart,
              'zstop': zstop,
              'zsamples': zsamples,
              'z': z,
              'r': r,
              'rstop': rstop,
              'rsamples': rsamples}

params_WA = {**pulse_params, **harm_params, **sol_params_WA}

single_func = functools.partial(calc_118_and_fluor,
                                zrange=zrange,
                                init_vals=[1e-10,1e-10,1e-10],
                                t_eval=z)

PAr_int = np.linspace(10,50,10)
PAr_scan = xr.DataArray(PAr_int,
                     dims = 'PAr',
                     attrs = {'units': 'Torr',
                              'long_name': "Partial Argon Pressure"})

scan_WA = scan_builder(single_func, params_WA, [PAr_scan])

scanned_WA = scan_WA(params=params_WA)





# sol_params_NA = {'func': curly_GBNA,
#               'initial_vals': (nonzero, nonzero),
#               'zstart': zstart,
#               'zstop': zstop,
#               'zsamples': zsamples,
#               'z': z,
#               'r': r,
#               'rstop': rstop,
#               'rsamples': rsamples}

# params_NA = {**pulse_params, **harm_params, **sol_params_NA}
# scan_NA = scan_builder(single_func, params_WA, [PXe_scan,dk_scan,b_scan])
# scanned_NA = scan_NA(params=params_NA)

np.save("dk_f_50cm_Pxe_3Torr.npy",scanned_WA.data)
# np.save("GBNA_Test.npy",scanned_NA.data)