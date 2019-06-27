
AVOGADRO = 6.0221e23 # mol^-1
BOLTZ = 1.3810e-23 # m^2*kg/s^2/K
EP0 = 8.8542e-12 # s^4*A^2/m^3/kg
CHARGE_E = 1.6022e-19 # Coulombs
GAS_CONSTANT = 8.3145 # J/mol/K
GRAV = 9.8067 # m/s^2
PLANCK = 6.6261e-34 # m^2*kg/s
EV_PER_JOULE = 6.2415e18

import numpy as np
  
def sumOverNs(*args):
    return tuple(np.squeeze(np.sum(arg,axis=0)) for arg in args)

def rename(x,*keys,dims='',axis=''):
    if axis == '':
        return tuple(x[key] for key in keys)
    return tuple(multiDim(x,key,dims,axis) for key in keys)
        
def multiDim(x,key,dims,axis):
    # Extend the given matrix to the Ns x Nt x Nv x Np baseline
    Ns,Nt,Nv,Np = dims
    
    if axis == 'Site (Maybe)':
        if key+'_std' in x:
            axis = 'Site'
            #np.random.seed(126)
            x[key] = np.abs(x[key]+x[key+'_std']*np.random.normal(0,1,Ns)).astype(np.float32)
        else:
            return x[key]
    
    if (type(x[key]) == int) or (type(x[key]) == float):
        return x[key] * np.ones([Ns,Nt,Nv,Np])
    
    if axis == 'V':
        m = np.repeat(x[key][:,np.newaxis], Ns, axis=1)
        m = np.repeat(m[:,:,np.newaxis], Nt, axis=2)
        m = np.repeat(m[:,:,:,np.newaxis], Np, axis=3)
        m = np.transpose(m, [1,2,0,3])
        
    elif axis == 'T':
        m = np.repeat(x[key][:,np.newaxis], Ns, axis=1)
        m = np.repeat(m[:,:,np.newaxis], Nv, axis=2)
        m = np.repeat(m[:,:,:,np.newaxis], Np, axis=3)
        m = np.transpose(m, [1,0,2,3])
        
    elif axis == 'P':
        m = np.repeat(x[key][:,np.newaxis], Ns, axis=1)
        m = np.repeat(m[:,:,np.newaxis], Nt, axis=2)
        m = np.repeat(m[:,:,:,np.newaxis], Nv, axis=3)
        m = np.transpose(m, [1,2,3,0])
            
    elif axis == 'Site':
        m = np.repeat(x[key][:,np.newaxis], Nt, axis=1)
        m = np.repeat(m[:,:,np.newaxis], Nv, axis=2)
        m = np.repeat(m[:,:,:,np.newaxis], Np, axis=3)
            
    else:
        raise ValueError('qsolve.py: multiDim() axis must be "V",' \
        '"T", "P", "Site" or "Site (Maybe)"')
    return m

def tooManySites(N, Nmax, fields, dims, metaTxt):
    if N > Nmax:
        zz = np.zeros(max(dims[1:])) # length of ind. variable (V, T or P)
        d = dict()
        for field in fields:
            d[field] = zz
        metaTxt += '<font color="red">ERROR: The field was not calculated because '\
        'the total number of sites is too high. The maximum value is %d</font><br>' % Nmax
        return d,metaTxt
    else:
        return None,metaTxt
            