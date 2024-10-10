import numpy as np
import beamFunctions as bf


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

def dA118_dz_GBNA(r,z0,z,amplitudes,params):
    chi3 = params['chi3']
    
    PXe = params['PXe'] * Torr_to_m3
    
    k118 = 2*np.pi/(118*10**(-9))
    
    [A_118, _] = amplitudes
    
    dA_118_dz = (1/2)*chi3*PXe*k118 * (bf.amplitude_355(r,z,params)**3) \
    * np.cos((2*z/b)-2*np.arctan2(2*z/b,1)+bf.phi3(z,z0,2/b,params))\
     -(bf.dBeam_radius_dz(z, params)/bf.beam_radius(z,params))*(1-((6*r**2)/(bf.beam_radius(z,params))**2))*A_118
   
    
    dA_fluo_dz = 0
    
    
    return [dA_118_dz, dA_fluo_dz]

def dA118_dz_GBWA(r,z0,z,amplitudes,params): #extra term to make it 3d

    
    chi3 = params['chi3']
    sigma = params['sigma']
    PXe = params['PXe'] * Torr_to_m3
    omega0 = params['omega0']
    #PXe = params['PXe'] * 3.54e16
    
    k118 = 2*np.pi/(118*10**(-9))
    
    [A_118, _] = amplitudes
    
    
    dA_118_dz = (1/2)*chi3*PXe*k118 * (amplitude_355(r,z,params)**3) \
    * np.cos((2*z/b)-2*np.arctan2(2*z/b,1)+phi3(z,z0,2/b,params))\
     -(dBeam_radius_dz(z, params)/beam_radius(z,params))*(1-((6*r**2)/(beam_radius(z,params))**2))*A_118 \
    -  0.5*sigma * PXe**2 * A_118
    
    dA_fluo_dz = 0.5*sigma * PXe**2 * A_118 
    
    
    return [dA_118_dz, dA_fluo_dz]

    
    
    
    