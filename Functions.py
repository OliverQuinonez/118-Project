import functools
import pandas as pd
import numpy as np
import tqdm
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as scint
xr.set_options(keep_attrs=True)

#########################################################################################################################

#Physical Constants
c = 299792458  # m/s 
kB = 1.38064852E-23  # (m^2 kg) / (s^2 K)
e_charge = 1.60217662E-19  # coulombs
eps_0 = 8.854187E-12  # F/m
m_electron = 9.109E-31  # kg
Torr_to_m3 = 3E22 #number density of gas per Torr (atoms/m^3)

#Common Values
lambda_355 = 355E-9  # m
lambda_118 = lambda_355/3  # m
def lambda_to_freq(lamb):
    return c/lamb
def freq_to_lambda(freq):
    return c/freq
omega_355 = lambda_to_freq(lambda_355)  # Hz
omega_118 = lambda_to_freq(lambda_355)  # Hz
mXe = 2.1801714E-25  # kg

nonzero = 1e-10

#########################################################################################################################

def beam_radius(z, params):
    omega0 = params['omega0']
    zR = params['zR']
    return omega0 * np.sqrt(1+(z/zR)**2)

def dBeam_radius_dz(z, params):
    omega0 = params['omega0']
    zR = params['zR']
    return omega0 * (z/(zR**2)) / np.sqrt(1+(z/zR)**2)

def peak_intensity_355(params):
    omega0 = params['omega0']
    energy = params['energy']
    duration = params['duration'] # [J] / (([m]^2) * [s]) = [W/m^2]
    return energy / ((np.pi * omega0**2) * duration)

def peak_amplitude_355(params):
    return np.sqrt(peak_intensity_355(params) / (2 * c * eps_0))
    
def amplitude_355(r,z, params):
    omega0 = params['omega0']
    return peak_amplitude_355(params) * (omega0) /(beam_radius(z, params))*np.exp(-r**2/beam_radius(z,params)**2)
            
def dAmplitude_355_dz(z, params):
    omega0 = params['omega0']
    return -peak_amplitude_355(params) *(omega0) * dBeam_radius_dz(z, params) / (beam_radius(z, params)**2) 

def ReJ3(z,z0,dk,params): # Returns the integral of the real part of J_3 from boyd 2.10.3
    #dk is phase mismatch, delta k
    b= params['b']
    Ref = lambda x: (1/(1+ (2*x/b)**2)**2) * ((1-(2*x/b)**2) *np.cos(dk*x) + (2*(2*x/b))*np.sin(dk*x))
    I = scint.quad(Ref,z0,z)
    #print('ReJ3Value: ', I[0], 'Error estimate: ', I[1])
    return I

def ImJ3(z,z0,dk,params):# Returns the integral of the real part of J_3 from boyd 2.10.3\
    #dk is phase mismatch, delta k
    b= params['b']
    Imf = lambda x: (1/(1+(2*x/b)**2)**2) * ((1-(2*x/b)**2)*np.sin(dk*x) - (2*(2*x/b))*np.cos(dk*x))
    I = scint.quad(Imf,z0,z)
    #print('ImJ3Value: ', I[0], 'Error estimate: ', I[1])
    return I
    
def phi3(z,z0,dk,params): #return the phase of the complex conjugate of J3
    return np.arctan2(-ImJ3(z,z0,dk,params)[0],ReJ3(z,z0,dk,params)[0])

#########################################################################################################################

def dA118_dz_GBNA(r,z0,z,amplitudes,params):
    chi3 = params['chi3']
    PXe = params['PXe'] * Torr_to_m3
    k118 = 2*np.pi/(118*10**(-9))
    b = params['b']
    
    [A_118, _] = amplitudes
    
    dA_118_dz = (1/2)*chi3*PXe*k118 * (amplitude_355(r,z,params)**3) \
    * np.cos((2*z/b)-2*np.arctan2(2*z/b,1)+phi3(z,z0,2/b,params))\
     -(dBeam_radius_dz(z, params)/beam_radius(z,params))*(1-((6*r**2)/(beam_radius(z,params))**2))*A_118
   
    dA_fluo_dz = 0
    
    return [dA_118_dz, dA_fluo_dz]

def dA118_dz_GBWA(r,z0,z,amplitudes,params): #extra term to make it 3d
    chi3 = params['chi3']
    sigma = params['sigma']
    PXe = params['PXe'] * Torr_to_m3
    omega0 = params['omega0']
    b = params['b']
    k118 = 2*np.pi/(118*10**(-9))
    
    [A_118, _] = amplitudes
    
    dA_118_dz = (1/2)*chi3*PXe*k118 * (amplitude_355(r,z,params)**3) \
    * np.cos((2*z/b)-2*np.arctan2(2*z/b,1)+phi3(z,z0,2/b,params))\
     -(dBeam_radius_dz(z, params)/beam_radius(z,params))*(1-((6*r**2)/(beam_radius(z,params))**2))*A_118 \
    -  0.5*sigma * PXe**2 * A_118
    
    dA_fluo_dz = 0.5*sigma * PXe**2 * A_118 
    
    return [dA_118_dz, dA_fluo_dz]

def solve_diff_eq(func, params, zrange, init_vals, z_eval,r_eval):
    
    beam_118_array = np.zeros((len(r_eval),len(z_eval)),np.longdouble)
    fluor_array = np.zeros((len(r_eval),len(z_eval)))
    arr_index = 0
    (zstart, zstop) = zrange
    
    for r in r_eval:
        sol = scint.solve_ivp(functools.partial(func, r, zstart+nonzero, params=params), 
                              zrange, init_vals, t_eval=z_eval,method ='Radau') # seems like r problem was stiff
        beam_118_array[arr_index] = sol['y'][0]
        fluor_array[arr_index] = sol['y'][1]
        arr_index += 1
           
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

#########################################################################################################################
def plot3dBeam(beam,r,z):
    Z,R = np.meshgrid(z,r)
    plt.contourf(Z,R,beam,64)

    zstart,zstop = (z[0],z[-1])
    rstop = r[-1]

    plt.ylim(0,rstop)
    plt.xlim(zstart,zstop)
    plt.xlabel('z (m)')
    plt.ylabel('r (m)')
    plt.title("$|A_{118}|$")
    plt.show()

def plot1dBeamR(beam,r,zindex):
    plt.plot(r,beam[:,zindex])
    plt.show()
    
       

    
    
    
    
    
    
    
    
    