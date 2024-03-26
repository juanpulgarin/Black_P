import numpy as np
import scipy.special as special


def spherical_solution(r,theta,phi,nr,l,m):

    Confluente_1_1 = special.gamma(nr)*special.gamma(l+0.5+1)/special.gamma(nr+l+0.5+1)*special.assoc_laguerre(r, nr, k=l+0.5)
    harmonics = special.sph_harm(m, l, theta, phi)
    
    return r**l*np.exp(-Lambda/2*r**2)*confluente_1_1*harmonics
