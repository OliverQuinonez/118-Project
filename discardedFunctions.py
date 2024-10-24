def ReJ3_gauss_quad(z,z0,dk,params): # Returns the integral of the real part of J_3 from boyd 2.10.3
    #dk is phase mismatch, delta k
    b= params['b']
    Ref1 = lambda x: (1/(1+ (2*x/b)**2)**2) * (1-(2*x/b)**2)* np.cos(dk*x) #times cos(dk*x)
    Ref2 = lambda x: (1/(1+ (2*x/b)**2)**2) * (2*(2*x/b))*np.sin(dk*x) #times sin(dk*x)                                 
    I1 = scint.quadrature(Ref1,z0,z, tol = 1e-10,rtol=1e-10,maxiter=500)
    I2 = scint.quadrature(Ref2,z0,z, tol = 1e-10,rtol=1e-10,maxiter=500)
    Itot = I1[0]+I2[0]
    return Itot

def ImJ3_gauss_quad(z,z0,dk,params):# Returns the integral of the real part of J_3 from boyd 2.10.3\
    #dk is phase mismatch, delta k
    b= params['b']
    Imf1 = lambda x: (1/(1+(2*x/b)**2)**2) * (1-(2*x/b)**2)*np.sin(dk*x) #times sin(dk*x)
    Imf2 = lambda x: (1/(1+(2*x/b)**2)**2)* (-(2*(2*x/b)))*np.cos(dk*x) #times *np.cos(dk*x)) 
    I1 = scint.quadrature(Imf1,z0,z, tol = 1e-10,rtol=1e-10,maxiter = 500)
    I2 = scint.quadrature(Imf2,z0,z, tol = 1e-10,rtol=1e-10,maxiter=500)
    Itot= I1[0]-I2[0]
    return Itot

def phi3_gauss_quad(z,z0,dk,params): #return the phase of the complex conjugate of J3
    return np.arctan2(-ImJ3_gauss_quad(z,z0,dk,params),ReJ3_gauss_quad(z,z0,dk,params))

def phi3_gauss_quad_arctan1(z,z0,dk,params):
    return np.arctan(-ImJ3_gauss_quad(z,z0,dk,params)/ReJ3_gauss_quad(z,z0,dk,params))

def phi3_gauss_quad_phase(z,z0,dk,params):
    w = np.complex(ReJ3_gauss_quad(z,z0,dk,params),-ImJ3_gauss_quad(z,z0,dk,params))
    return np.phase(w)


def phi3(z,z0,dk,params): #return the phase of the complex conjugate of J3
    J = complex(ReJ3,ImJ3)
    return -cmath.phase(J)

def dA118_dz_GBNA(r,Phi,z0,z,amplitudes,params):
    chi3 = params['chi3']
    PXe = params['PXe'] * Torr_to_m3
    k118 = 2*np.pi/(118*10**(-9))
    b = params['b']
    
    [A_118, _] = amplitudes



    dA_118_dz = (1/2)*chi3*PXe*k118 * (amplitude_355(r,z,params)**3) \
    * np.cos((2*z/b)-2*np.arctan2(2*z/b,1) + phi3_interp(z,Phi,params))\
     +(dBeam_radius_dz(z, params)/beam_radius(z,params))*(((6*r**2)/(beam_radius(z,params))**2)-1)*A_118
   
    dA_fluo_dz = 0
    
    return [dA_118_dz, dA_fluo_dz]

def dA118_dz_GBWA(r,z0,z,amplitudes,params): #extra term to make it 3d
    chi3 = params['chi3']
    sigma = params['sigma']
    PXe = params['PXe'] * Torr_to_m3
    omega0 = params['omega0']
    b = params['b']
    zsamples = params['zsamples']
    k118 = 2*np.pi/(118*10**(-9))
    
    [A_118, _] = amplitudes

    dA_118_dz = (1/2)*chi3*PXe*k118 * (amplitude_355(r,z,params)**3) \
    * np.cos((2*z/b)-2*np.arctan2(2*z/b,1)+phi3_gauss_quad_complex(z,z0,2/b,params))\
     -(dBeam_radius_dz(z, params)/beam_radius(z,params))*(1-((6*r**2)/(beam_radius(z,params))**2))*A_118 \
    -  0.5*sigma * PXe**2 * A_118
    
    dA_fluo_dz = 0.5*sigma * PXe**2 * A_118 
    
    return [dA_118_dz, dA_fluo_dz]


def solve_diff_eq_3D(func, params, zrange, init_vals, z_eval,r_eval):

    b = params['b']
    z = params['z']
    zsamples = params['zsamples']
    zstart = params['zstart']

    Phi = np.zeros(len(z))
    i =0
    for zval in z:
        Phi[i] = evalPhi3(zval, zstart, 2/b, params) #precomputes Phi 3 (interpolated in diff eq)
        i += 1

    beam_118_array = np.zeros((len(r_eval),len(z_eval)),np.longdouble)
    fluor_array = np.zeros((len(r_eval),len(z_eval)))
    arr_index = 0
    (zstart, zstop) = zrange
    for r in r_eval:
        sol = scint.solve_ivp(functools.partial(func, r, Phi,zstart+nonzero,params=params), 
                            zrange, init_vals, t_eval=z_eval,method ='Radau',rtol = 1e-5,atol = 1e-10)
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

