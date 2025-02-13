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
from Functions import *


omega0 = 0.5e-2
b=k355*omega0**2
zR = b / 2     # [m] Rayleigh Range




pulse_params = {'b' : b,            # [m] confocal parameter
            'energy' : 15e-3,       # [J] single pulse energy original
            'duration' : 7e-9,       # [s] single pulse length
            'PXe':10,
            'dk':0}    
harm_params = {'alpha' : 8.20839154e-48, #value from fit 'alpha' : 8.20839154e-48,
            'chi3' : 1.5e-35}   #value from fit: 1.5e-35

zstart = -0.2
zstop =0.2
zsamples = 1000
zrange = (zstart, zstop)
z = np.array(np.linspace(zstart, zstop, zsamples,dtype = np.longdouble))

rstop = 10*omega0
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

bsamples = 10
brange = (f_to_b(omega0,20e-2),f_to_b(omega0,100e-2))# confocal perameter [m] (corresponds to f=20cm to f=100cm)
b_int = np.linspace(brange[0], brange[1], bsamples)
bscan = xr.DataArray(b_int,
                     dims = 'b',
                     attrs = {'units': 'm',
                              'long_name': "Confocal parameter"})
single_func = functools.partial(calc_118_and_fluor,
                                zrange=zrange,
                                init_vals=[1e-10,1e-10,1e-10],
                                t_eval=z)

dk_samples = bsamples
dk =np.linspace(PAr_to_dk(0,params_WA['PXe']),PAr_to_dk(1500,params_WA['PXe']),dk_samples)
dk_scan = xr.DataArray(dk,
                     dims = 'dk',
                     attrs = {'units': 'm^-1',
                              'long_name': "Phase Mismatch"})

B_scan = scan_builder(single_func, params_WA, [bscan,dk_scan])

scanned_WA = B_scan(params=params_WA)





sol_params_NA = {'func': curly_GBNA,
              'initial_vals': (nonzero, nonzero),
              'zstart': zstart,
              'zstop': zstop,
              'zsamples': zsamples,
              'z': z,
              'r': r,
              'rstop': rstop,
              'rsamples': rsamples}

params_NA = {**pulse_params, **harm_params, **sol_params_NA}

scan_NA = scan_builder(single_func, params_NA, [bscan,dk_scan])
scanned_NA = scan_NA(params=params_NA)


np.save("dksamples=10_GBWA_E=150mj_PXe="+str(pulse_params["PXe"])+"torr_fstart=20cm_fstop=100cm.npy",scanned_WA.data)
np.save("dksamples=10_GBNA_E=150mj_PXe="+str(pulse_params["PXe"])+"torr_fstart=20cm_bstop=100cm.npy",scanned_NA.data)