import functools
import pandas as pd
import numpy as np
import tqdm
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as scint


import diffEqs as de
import beamFunctions as bf

global c,kB,e_charge, eps_0, m_electron, Torr_to_m3, omega_355, omega_118, m_electron, r_int,b,zR,omega0,z,nonzero


c = 299792458  # m/s 
kB = 1.38064852E-23  # (m^2 kg) / (s^2 K)
e_charge = 1.60217662E-19  # coulombs
eps_0 = 8.854187E-12  # F/m
m_electron = 9.109E-31  # kg
Torr_to_m3 = 3E22 #number density of gas per Torr (atoms/m^3)
nonzero = 1E-10

lambda_355 = 355E-9  # m
lambda_118 = lambda_355/3  # m
def lambda_to_freq(lamb):
    return c/lamb

def freq_to_lambda(freq):
    return c/freq
omega_355 = lambda_to_freq(lambda_355)  # Hz
omega_118 = lambda_to_freq(lambda_355)  # Hz

mXe = 2.1801714E-25  # kg

zstart = -0.2
#zstart = nonzero

zstop = 0.2
#zstop = nonzero

zrange = (zstart, zstop)
plotrange = (-0.08, 0.08)
z = np.array(np.linspace(zstart, zstop, 1000))
dz = z[1] - z[0]

b = 0.023   
zR = b / 2  
omega0 = np.sqrt(lambda_355 * zR / np.pi) # [m] Beam radius at the focus
rstop = 8*omega0
r_int = np.linspace(0,rstop,100)
def driver():
    func = de.dA118_dz_GBNA

    PXe = 25

    chi2 = 1.5e-50 # [Units?] value from johns file

    chi3 = 1.5e-35   # [Units?] value from johns file

    sigma_1p1 = 6.3E-23

    sigma_1p1_355 = 1.6E-23


    pulse_params = {'b' : b,            # [m] confocal parameter
                    'zR' : zR,           # [m] Rayleigh range
                    'omega0' : omega0,   # [m] beam waist at focus
                    'energy' : 0.017,       # [J] single pulse energy
                    'duration' : 7e-9}     # [s] single pulse length

    harm_params = {'sigma' : chi2,
                'chi3' : chi3,
                'PXe' : PXe,
                'sigma_1p1' : sigma_1p1,
                'sigma_1p1_355' : sigma_1p1_355
                }
    nonzero = 1e-10
    params = {**pulse_params, **harm_params}
    #initial_vals = (nonzero, nonzero, nonzero)
    initial_vals = (nonzero, nonzero)





    data = solve_diff_eq(func, params, zrange, initial_vals, z)

    dat = [data.beam_118]
    mag2 = np.array(np.zeros(len(z))) # values of the magnitude of J_3 for plotting, similar to phi3


    index=0
    for num in z:
        mag2[index] = np.sqrt(bf.ReJ3(num,zstart,2/b,params)[0]**2+bf.ImJ3(num,zstart,2/b,params)[0]**2)
        index+=1
    #print(mag2)
    # plt.plot(z,mag2)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)


    # for d in dat:

    #     ax.plot(d.z, d[0]/max(d[0])), 
    #     #ax.plot(d.z, d[0]/max(d[0])), 
    #     #label=f"{d.attrs['long_name']}. Peak value: {max(d[500]).data:.2G}"
        
    # ax.set_xlim([zstart, zstop])
    # ax.set_xlabel('z position (m)')
    # ax.set_ylabel('Amplitude (arb.)')

    # k118 = 2*np.pi/(118*10**(-9))


    # boyd118OnAxis =(1/2) * k118 * chi3 * Torr_to_m3 * params['PXe'] * bf.peak_amplitude_355(params)**3 \
    # *mag2\
    # *omega0/bf.beam_radius(z,params)*np.exp(-3*(0*omega0)**2/bf.beam_radius(z,params)**2)

    # plt.plot(z,boyd118OnAxis/max(boyd118OnAxis))
    # plt.legend(['Numerical','Boyd'])
    # plt.xlim(zstart,zstop)
    # plt.show()


        
    Z,R = np.meshgrid(z,r_int)

    #full_beam_355 = np.zeroes((len(r),len(z)))

    # full_beam_355 = amplitude_355(Z,params)*np.exp(-(Y**2)/(beam_radius(Z,params)**2))
    # full_beam_118 = np.tile(data.beam_118,(len(Z),1))*np.exp(-(3*Y**2)/(beam_radius(Z,params)**2))

    # plt.contourf(Z,R,bf.amplitude_355(R,Z, params),64)
    # plt.ylim(0,rstop)
    # plt.xlim(zstart,zstop)
    # plt.xlabel('z distance (m)')
    # plt.ylabel('r distance (m)')
    # plt.title('$|E_{355}|$')
    # plt.show()

    plt.contourf(Z,R,data.beam_118,64)
    plt.ylim(0,rstop)
    plt.xlim(zstart,zstop)
    plt.xlabel('z distance (m)')
    plt.ylabel('y distance (m)')
    plt.title("$|E_{118}|$")
    plt.show()

    # plt.contourf(Z,R,data.fluor,64)
    # plt.ylim(-rstop,rstop)
    # plt.xlim(zstart,zstop)
    # plt.xlabel('z distance (m)')
    # plt.ylabel('y distance (m)')
    # plt.title("Fluorescence")
    # plt.show()

    k118 = 2*np.pi/(118*10**(-9))
    boyd118 = (1/2) * k118 * chi3 * Torr_to_m3 *params['PXe'] * bf.peak_amplitude_355(params) **3 \
    * mag2 \
    * omega0/bf.beam_radius(Z,params)*np.exp(-3*R**2/bf.beam_radius(Z,params)**2)

    # plt.contourf(Z,R,boyd118,64)
    # plt.ylim(0,rstop)
    # plt.xlim(zstart,zstop)
    # plt.show()

    # err = abs(data.beam_118 - boyd118)
    # plt.contourf(Z,R,err)

    plt.show()


plt.show()   


def solve_diff_eq(func, params, zrange,init_vals, t_eval):
    
    beam_118_array = np.zeros((len(r_int),len(z)))
    fluor_array = np.zeros((len(r_int),len(z)))
    arr_index = 0
    
    
    for r in r_int:

        sol = scint.solve_ivp(functools.partial(func, r, zstart+nonzero, params=params), 
                              zrange, init_vals, t_eval=t_eval, method = 'Radau')
        
        
        beam_118_array[arr_index] = sol['y'][0]
        fluor_array[arr_index] = sol['y'][1]


        arr_index += 1

    
#     #package into an xarray
#     beam_355 = xr.DataArray(, 
#                             dims = ('z'), 
#                             coords = {'z': sol['t']},
#                             attrs = {'units': 'V/m',
#                             'long_name': "355 nm amplitude"})
                    
    beam_118 = xr.DataArray(beam_118_array, 
                            dims = ('r','z'), 
                            coords = {'z': sol['t']},
                            attrs = {'units': 'V/m',
                                     'long_name': "118 nm amplitude"})
    fluor = xr.DataArray(fluor_array, 
                            dims = ('r','z'), 
                            coords = {'z': sol['t']},
                            attrs = {'units': 'arb.',
                                     'long_name': "Fluorescence"})
    
    data = xr.Dataset(
        data_vars = {#'beam_355': beam_355,
                     'beam_118': beam_118,
                     'fluor': fluor},
        attrs = {'z_range': zrange, 'init_vals': init_vals, **params}
        )
    return data
    




driver()