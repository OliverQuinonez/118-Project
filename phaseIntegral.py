import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

import beamFunctions as bf
import helperFunctions as hf




def driver():

    lambda_355 = 355E-9  # m
    lambda_118 = lambda_355/3  # m



    omega_355 = lambda_to_freq(lambda_355)  # Hz
    omega_118 = lambda_to_freq(lambda_355)  # Hz

    mXe = 2.1801714E-25  # kg

    b = 0.023      # [m] Confocal parameter taken from 118 OH paper
    zR = b / 2     # [m] Rayleigh Range
    omega0 = np.sqrt(lambda_355 * zR / np.pi) # [m] Beam radius at the focus


    dk = 2/b

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

    Ref = lambda x: (1/(1+ (2*x/b)**2)**2) * ((1-(2*x/b)**2) *np.cos(dk*x) + (2*(2*x/b))*np.sin(dk*x))
    Imf = lambda x: (1/(1+(2*x/b)**2)**2) * ((1-(2*x/b)**2)*np.sin(dk*x) - (2*(2*x/b))*np.cos(dk*x))


    z = np.linspace(-0.2,0.2,1000)

    # plt.plot(z,Ref(z))
    # plt.plot(z,Imf(z))

    ReJ3_Val = np.zeros(len(z))
    index=0
    for num in z:
        ReJ3_Val[index] = bf.ReJ3(num,-0.2,dk,params)[0]
        index+=1

    ImJ3_Val = np.zeros(len(z))
    index=0
    for num in z:
        ImJ3_Val[index] = bf.ImJ3(num,-0.2,dk,params)[0]
        index+=1

    plt.plot(z,bf.phi3(z,-0.2,dk,params))
    plt.show()




def lambda_to_freq(lamb):
    c = 299792458  # m/s 
    return c/lamb

def freq_to_lambda(freq):
    c = 299792458  # m/s 
    return c/freq

driver()
