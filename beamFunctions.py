import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

c = 299792458  # m/s 
kB = 1.38064852E-23  # (m^2 kg) / (s^2 K)
e_charge = 1.60217662E-19  # coulombs
eps_0 = 8.854187E-12  # F/m
m_electron = 9.109E-31  # kg
Torr_to_m3 = 3E22 #number density of gas per Torr (atoms/m^3)

lambda_355 = 355E-9  # m
lambda_118 = lambda_355/3  # m

def lambda_to_freq(lamb):
    return c/lamb

def freq_to_lambda(freq):
    return c/freq

omega_355 = lambda_to_freq(lambda_355)  # Hz
omega_118 = lambda_to_freq(lambda_355)  # Hz

mXe = 2.1801714E-25  # kg

b = 0.023      # [m] Confocal parameter taken from 118 OH paper
zR = b / 2     # [m] Rayleigh Range
omega0 = np.sqrt(lambda_355 * zR / np.pi) # [m] Beam radius at the focus

pulse_params = {'b' : b,            # [m] confocal parameter
                   'zR' : zR,           # [m] Rayleigh range
                   'omega0' : omega0,   # [m] beam waist at focus
                   'energy' : 0.017,       # [J] single pulse energy
                   'duration' : 7e-9}     # [s] single pulse length

harm_params = {#'sigma' : chi2,
               #'chi3' : chi3,
               #'PXe' : PXe,
               #'sigma_1p1' : sigma_1p1,
               #'sigma_1p1_355' : sigma_1p1_355
               }

params = {**pulse_params, **harm_params}

def beam_radius(z, params):
    
    omega0 = params['omega0']
    zR = params['zR']
    # [m] * sqrt(1 + ([m]/[m])^2) = [m]
    return omega0 * np.sqrt(1+(z/zR)**2)

def dBeam_radius_dz(z, params):
  
    
    omega0 = params['omega0']
    zR = params['zR']
    # [m] * ([m] / [m]^2) / sqrt(1 + ([m]/[m])^2) = 1 ([m]/[m])
    return omega0 * (z/(zR**2)) / np.sqrt(1+(z/zR)**2)

def peak_intensity_355(params):
    
    
    omega0 = params['omega0']
    energy = params['energy']
    duration = params['duration']
    # [J] / (([m]^2) * [s]) = [W/m^2]
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
    #Ref = lambda x:  ((1/(1+ (2*x/b)**2))*np.cos(dk*x) + (2*(2*x/b)/(1+ (2*x/b)**2)**2)*np.sin(dk*x))
    
    #x is dummy integration variable
    return scint.quad(Ref,z0,z)

def ImJ3(z,z0,dk,params):
    #dk is phase mismatch, delta k
    
    b= params['b']
    
    Imf = lambda x: (1/(1+(2*x/b)**2)**2) * ((1-(2*x/b)**2)*np.sin(dk*x) - (2*(2*x/b))*np.cos(dk*x))
    #Imf = lambda x: ((1/(1+(2*x/b)**2))*np.sin(dk*x) - (2*(2*x/b)/(1+(2*x/b)**2)**2)*np.cos(dk*x))
    
    
    return scint.quad(Imf,z0,z)
    # Returns the integral of the real part of J_3 from boyd 2.10.3

def phi3(z,z0,dk,params): #return the phase of the complex conjugate of J3
    return np.arctan2(-ImJ3(z,z0,dk,params)[0],ReJ3(z,z0,dk,params)[0])

#def IntegralToArray()
    
    
    
    
    