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
import scipy

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


k118 = 2*np.pi/(118e-9)
k355 = 2*np.pi/(355e-9)

nonzero = 1e-10


#########################################################################################################################

def evalOmega0(b):
    return np.sqrt(lambda_355 * b / (2* np.pi))

def evalDeltaK(b):
    return 2/b

def beam_radius(z, params):
    b = params['b']
    zR = b/2
    omega0 = evalOmega0(b)
    return omega0 * np.sqrt(1+(z/zR)**2)

def dBeam_radius_dz(z, params):
    b = params['b']
    omega0 = evalOmega0(b)
    zR = b/2
    return omega0 * (z/(zR**2)) / np.sqrt(1+(z/zR)**2)

def peak_intensity_355(params):
    b = params['b']
    omega0 = evalOmega0(b)
    energy = params['energy']
    duration = params['duration'] # [J] / (([m]^2) * [s]) = [W/m^2]
    return energy / ((np.pi * omega0**2) * duration)

def peak_amplitude_355(params):
    return np.sqrt(peak_intensity_355(params) / (2 * c * eps_0))
    
def amplitude_355(r,z, params):
    b = params['b']
    omega0 = evalOmega0(b)
    return peak_amplitude_355(params) * (omega0) /(beam_radius(z, params))*np.exp(-r**2/beam_radius(z,params)**2)
            
def dAmplitude_355_dz(z, params):
    b = params['b']
    omega0 = evalOmega0(b)
    return -peak_amplitude_355(params) *(omega0) * dBeam_radius_dz(z, params) / (beam_radius(z, params)**2) 

def evalRef(x,params):
    b= params['b']
    PXe = params['PXe']
    PAr =params['PAr']
    dk = PAr_to_dk(PAr,PXe)
    return (1/(1+ (2*x/b)**2)**2) * (1-(2*x/b)**2)* np.cos(dk*x)+(1/(1+ (2*x/b)**2)**2) * (2*(2*x/b))*np.sin(dk*x)
def evalImf(x,params):
    b= params['b']
    PXe = params['PXe']
    PAr =params['PAr']
    dk = PAr_to_dk(PAr,PXe)
    return (1/(1+(2*x/b)**2)**2) * (1-(2*x/b)**2)*np.sin(dk*x) - (1/(1+(2*x/b)**2)**2)* (-(2*(2*x/b)))*np.cos(dk*x)

def ReJ3(z,z0,params): # Returns the integral of the real part of J_3 from boyd 2.10.3
    #dk is phase mismatch, delta k
    b= params['b']
    PXe = params['PXe']
    PAr =params['PAr']
    dk = PAr_to_dk(PAr,PXe)
    Ref1 = lambda x: (1/(1+ (2*x/b)**2)**2) * (1-(2*x/b)**2) #times cos(dk*x)
    Ref2 = lambda x: (1/(1+ (2*x/b)**2)**2) * (2*(2*x/b)) #times sin(dk*x)                                 
    I1 = scint.quad(Ref1,z0,z,weight ='cos',wvar = dk,limit =10000)
    I2 = scint.quad(Ref2,z0,z, weight = 'sin', wvar =dk, limit = 10000)
    Itot = I1[0]+I2[0]
    return Itot

def ImJ3(z,z0,params):# Returns the integral of the real part of J_3 from boyd 2.10.3\
    #dk is phase mismatch, delta k
    b= params['b']
    PXe = params['PXe']
    PAr =params['PAr']
    dk = PAr_to_dk(PAr,PXe)
    Imf1 = lambda x: (1/(1+(2*x/b)**2)**2) * (1-(2*x/b)**2) #times sin(dk*x)
    Imf2 = lambda x: (1/(1+(2*x/b)**2)**2)* (-(2*(2*x/b))) #times *np.cos(dk*x)) 
    I1 = scint.quad(Imf1,z0,z, weight = 'sin', wvar = dk,limit=10000)
    I2 = scint.quad(Imf2,z0,z, weight = 'cos', wvar = dk,limit = 10000)
    Itot= I1[0]-I2[0]
    return Itot

def evalPhi3(z,z0,params):
    #Calculates the phase of J_3 conjugate from Boyd 2.10.3
    #uses Gaussian quadrature
    b = params['b']
    PXe = params['PXe']
    PAr =params['PAr']
    dk = PAr_to_dk(PAr,PXe)
    #complex_f = lambda x: (np.cos(x*dk) + 1j*np.sin(x*dk))/(1+1j*(2*x/b))**2

    #def Ref(x):
        #return complex_f(x).real
    #def Imf(x):
        #return complex_f(x).imag

    #ReJ = scint.quad(Ref,z0,z,limit =10000)[0]
    #ImJ = scint.quad(Imf,z0,z,limit = 10000 )[0]
    ReJ = ReJ3(z,z0,params)
    ImJ = ImJ3(z,z0,params)

    J = complex(ReJ,ImJ)
    return -cmath.phase(J)

def phi3_interp(zval,Phi,params):
    #Interpolating function for the value of the integral J3 at any point
    z = params['z']
    phi3_interp = scinterp.interp1d(z, Phi, kind='cubic', fill_value='extrapolate')
    return phi3_interp(zval)


#########################################################################################################################



def curly_GBNA(z0,z,amplitudes,params):
    chi3 = params['chi3']
    PXe =params['PXe']
    NXe =PXe  * Torr_to_m3
    PAr = params['PAr']

    k118 = 2*np.pi/(118*10**(-9))
    b = params['b']
    dk = PAr_to_dk(PAr,PXe)
    omega0 = evalOmega0(b)
    
    [A_118, _] = amplitudes

    dA_118_dz = (1/2)*chi3*NXe*k118 * (omega0/beam_radius(z,params))**2 * (peak_amplitude_355(params)**3) \
    * np.cos(dk*z-2*np.arctan2(2*z/b,1) + evalPhi3(z,z0,params))\
   
    dA_fluo_dz = 0
    
    return [dA_118_dz, dA_fluo_dz]

def curly_GBWA(z0,z,amplitudes,params):
    chi3 = params['chi3']
    PXe =params['PXe']
    NXe =PXe  * Torr_to_m3
    PAr = params['PAr']

    k118 = 2*np.pi/(118*10**(-9))
    b = params['b']
    dk = PAr_to_dk(PAr,PXe)
    omega0 = evalOmega0(b)

    alpha = params['alpha']

    [A_118, _] = amplitudes

    dA_118_dz = (1/2)*chi3*NXe*k118 * (omega0/beam_radius(z,params))**2 * (peak_amplitude_355(params)**3) \
    * np.cos((dk*z)-2*np.arctan2(2*z/b,1) + evalPhi3(z,z0,params))\
    -(1/2) * alpha* NXe**2 * A_118

    dA_fluo_dz = (1/2) * alpha* NXe**2 * A_118
    
    return [dA_118_dz, dA_fluo_dz]

def John_WA(z0,z, amplitudes, params):
   
    chi3 = params['chi3']
    alpha = params['alpha']
    PXe = params['PXe'] * Torr_to_m3
    #PXe = params['PXe'] * 3.54e16
    
    k118 = 2*np.pi/(118*10**(-9))
    
    [A_118, _] = amplitudes
    #dA_355_dz = dAmplitude_355_dz(z, params)
    dA_118_dz = chi3*PXe*k118 * amplitude_355(0,z,params)**3 - 0.5*alpha * PXe**2 * A_118
    dA_fluo_dz = 0
    
    return [dA_118_dz, dA_fluo_dz]

def John_NA(z0,z, amplitudes, params):
   
    chi3 = params['chi3']
    alpha = params['alpha']
    PXe = params['PXe'] * Torr_to_m3
    #PXe = params['PXe'] * 3.54e16
    
    k118 = 2*np.pi/(118*10**(-9))
    
    [A_118, _] = amplitudes
    #dA_355_dz = dAmplitude_355_dz(z, params)
    dA_118_dz = chi3*PXe*k118 * amplitude_355(0,z,params)**3
    dA_fluo_dz = 0
    
    return [dA_118_dz, dA_fluo_dz]








def solve_diff_eq(func, params, zrange, init_vals, z_eval,r_eval):

    z = params['z']
    r = params['r']
    zsamples = params['zsamples']
    rsamples = params['rsamples']
    zstart = params['zstart']
    b=params['b']
    omega0 = evalOmega0(b)

    # Phi = np.zeros(len(z))
    # i =0
    # for zval in z:
    #     Phi[i] = evalPhi3(zval, zstart, params) #precomputes Phi 3 (interpolated in diff eq)
    #     i += 1
    # print('b:',b)
    # print('dk:',evalDeltaK(params))
    # print('w0:', evalOmega0(params))
    arr_index = 0
    (zstart, zstop) = zrange

    print("Function: ",func," (PXe,PAr,f)"," (", params['PXe']," , ",params['PAr'], " , ", b_to_f(0.5e-2,params["b"])," )")
    #print(dk_to_PAr(dk))

    sol = scint.solve_ivp(functools.partial(func,zstart+nonzero,params=params), 
                        zrange, init_vals, t_eval=z_eval,method ='Radau',atol=10e-8,rtol=1e-4)
    
    
    #beam_118_array = np.zeros((rsamples,zsamples),np.longdouble)#initalize beam and flourescene arrays
    fluor_array = np.zeros(zsamples,np.longdouble)



    # arr_index = 0
    # for rval in r:
    #     beam_118_array[arr_index,:] = sol['y'][0]*(omega0/beam_radius(z,params))*np.exp(-3*rval**2/beam_radius(z,params)**2)
    #     fluor_array[arr_index] = sol['y'][1]*(omega0/beam_radius(z,params))*np.exp(-3*rval**2/beam_radius(z,params)**2)
    #     arr_index += 1

    
    
           
    beam_118 = xr.DataArray(sol['y'][0], 
                            #dims = ('r','z'), 
                            #coords = {'z': sol['t']},
                            attrs = {'units': 'V/m',
                                     'long_name': "118 nm amplitude"})
    fluor = xr.DataArray(fluor_array, 
                            #dims = ('r','z'), 
                            #coords = {'z': sol['t']},
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


def scan_parameter_1D_ODE(param_scan_func, params, scan_range):
    """
    param_scan_func: A function of the form 'func(params)'. The function
        should return a well-formed DataArray
    scan_range: DataArray of scan variable
    
    
    """
    
    scan_in_params = scan_range.dims[0] in params
    if not scan_in_params:
        raise KeyError("Scan parameter is not in parameter list.")
        
    new_data = []    
    for param_val in scan_range.data:
        # print('scan_range.dims[0]: ',scan_range.dims[0])
        # print('params[scan_range.dims[0]] before: ',params[scan_range.dims[0]])
        params[scan_range.dims[0]] = param_val
        # print('params[scan_range.dims[0]] after: ',params[scan_range.dims[0]])
        new_data.append(param_scan_func(params=params))
        # print(new_data)
    
    old_coords = list(new_data[0].dims)
    new_coords = [scan_range] + [new_data[0].coords[key] for key in old_coords]
    new_xr = xr.DataArray(new_data, coords=new_coords)
    
    scanned_params = old_coords + [scan_range.dims[0]]
    unscanned_params = {key: params[key] 
                        for key in params 
                        if key not in scanned_params}
    
    new_xr.attrs = {**unscanned_params, **new_data[0].attrs}
    
    return new_xr

def calc_118_and_fluor(params, zrange, init_vals, t_eval):
    """
    
    
    """
    func = params["func"]
    b = params['b']
    omega0 = evalOmega0(b)
    zstop = params['zstop']
    initial_vals = params['initial_vals']
    z = params['z']
    r = params['r']
#     print("params[func]:",params["func"])
#     print("diff_eqs[params[func]]:",diff_eqs[params["func"]])
#     print("func:", func)
    sol = solve_diff_eq(func,params, zrange, initial_vals, z,r)
    
    if(func == "John_WA" or func == "John_NA"):
        detected_118 = sol.beam_118[-1]**2

    else:
        detected_118 = (1/4)*np.sqrt((2*np.pi)/3) * (sol.beam_118[-1])**2 * (omega0**2/beam_radius(zstop,params))
    
    #fluor_low, fluor_high = params['fluor_detect_window']
    #in_window = ((t_eval>=fluor_low) * (t_eval<=fluor_high)).astype(int)
    #fluorescence = sum((data.fluor*in_window)**2)

    
    # result = xr.DataArray([detected_118.data, fluorescence.data],
    #                      dims='variable',
    #                      coords={'variable': ['118 signal', 'Fluorescence signal']},
    #                      attrs={'units': 'Arb.',
    #                             'zrange': zrange,
    #                             'init_vals': init_vals,
    #                             't_eval': t_eval,
    #                             'func': func.__name__})

    fluorescence = 0
    # print('detected_118.data: ',detected_118)
    # print('detected_118.data type: ',type(detected_118))
    result = xr.DataArray([detected_118.data, fluorescence],
                        dims='variable',
                        coords={'variable': ['118 signal', 'Fluorescence signal']},
                        attrs={'units': 'Arb.',
                            'zrange': zrange,
                            'init_vals': init_vals,
                            't_eval': t_eval,
                            'func': func.__name__})
    return result
    


def scan_builder(func, params, scans):
    """
    func: func(params)
    params: dict of params
    scans: list of DataArrays to scan.
    
    """
    
    for scan in scans:
        func = functools.partial(scan_parameter_1D_ODE,
                                 param_scan_func=func,
                                 scan_range=scan)
    return func   

def norm_DataArray(dataarray, val=None):
    """
    Normalize the input dataarray against itself, or to another value if specified.
    """
    if val is None:
        return dataarray / np.max(dataarray)
    else:
        return dataarray / val

def plot_pressure_scan(data, ax, selections=None, norm=True):
    sel = {**selections}
    if norm:
        #norm_DataArray(data.sel(sel)).plot(ax=ax, label=data.attrs["tlabel"])
        norm_DataArray(data.sel(sel)).plot(ax=ax)
    else:
        #data.sel(sel).plot(ax=ax, label=data.attrs["tlabel"])
        data.sel(sel).plot(ax=ax)
    ax.set_title(f"Scanning: Pressure")
    ax.legend()
       
##################################################################################

def free_space(q1,z):
    return q1+z

def thin_lens(q1,f):
    q2 = 1/((1/q1)-(1/f))
    return q2

def q_to_params(q,params):
    wavelength = lambda_355
    q_inv = 1/q
    Im_q_inv = q_inv.imag
    Re_q = q.real
    omega_final = np.sqrt(-wavelength/(np.pi*Im_q_inv))
    focus_final = Re_q
    

    print('positon relative to focus: ',focus_final,'[m]')
    print("beam spot size: ", omega_final,'[m]')

def f_to_b(omega0_initial,f):
    wavelength = lambda_355

    b_initial = (2*np.pi/ lambda_355)*omega0_initial**2

    b_final = 2*lambda_355*f**2/(np.pi*omega0_initial**2)*(1+(2*f/b_initial)**2)

    return b_final

def b_to_f(omega0_initial,b_final):
    wavelength = lambda_355
    b_initial = (2*np.pi/ lambda_355)*omega0_initial**2
    r = np.sqrt( (2*wavelength/(np.pi*omega0_initial**2))**2 + 4*(2/b_initial)**2 * (2*wavelength/(np.pi*omega0_initial**2) * b_final)) 

    n = -2*wavelength/(np.pi*omega0_initial**2)+r

    frac = n/(2*(2/b_initial)**2 * (2*wavelength/(np.pi*omega0_initial**2)))

    return np.sqrt(frac)

def dk_to_PAr(dk,PXe):
    CXe= -6.12E-21 #m^2
    CAr = 5.33E-22 #m^2
    NXe= PXe*Torr_to_m3
    NAr = (dk - NXe*CXe)/CAr
    return NAr/Torr_to_m3

def PAr_to_dk(PAr,PXe):
    CXe= -6.12E-21 #m^2
    CAr = 5.33E-22 #m^2
    NAr = PAr*Torr_to_m3
    NXe = PXe*Torr_to_m3

    dk = CXe*NXe+CAr*NAr

    return dk
    
    
    
    
    
    
    
    
    