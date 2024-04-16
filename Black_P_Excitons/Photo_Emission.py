import numpy as np

def Delta (E,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2)

def SHIFTING_KplusP(nk1,nk2,nk3,Ψ_k,pexc):
    Ψ = np.reshape(Ψ_k, [nk1,nk2,nk3])

    Ψ_r = np.fft.fftn(Ψ)

    xr = np.fft.fftfreq(nk1,d=1. / nk1).astype(int)
    yr = np.fft.fftfreq(nk2,d=1. / nk2).astype(int)
    zr = np.fft.fftfreq(nk3,d=1. / nk3).astype(int)

    ex1 = np.exp(-2.0j*np.pi*xr*(pexc[0]))
    ex2 = np.exp(-2.0j*np.pi*yr*(pexc[1]))
    ex3 = np.exp(-2.0j*np.pi*zr*(pexc[2]))

    Ψ_r_p = np.einsum('i,j,k,ijk->ijk', ex1, ex2,ex3, Ψ_r)

    Ψ_k_p = np.fft.ifftn(Ψ_r_p)

    return Ψ_k_p

def FourierInterpolate(nk1,nk2,nk3,Ψ_k_p,kpts):
    # x = np.reshape(vect[ist,:], [nk1,nk2])
    Ψ = np.reshape(Ψ_k_p, [nk1,nk2,nk3])
    Ψ_r = np.fft.fftn(Ψ)

    xr = np.fft.fftfreq(nk1,d=1. / nk1).astype(int)
    yr = np.fft.fftfreq(nk2,d=1. / nk2).astype(int)
    zr = np.fft.fftfreq(nk3,d=1. / nk3).astype(int)

    nk = kpts.shape[0]

    vect_int = np.zeros(nk,dtype=np.complex_)

    for ik in range(nk):
        kpt = kpts[ik,:]

        ex1 = np.exp(2.0j*np.pi*xr*(kpt[0]+0.5))
        ex2 = np.exp(2.0j*np.pi*yr*(kpt[1]+0.5))
        ex3 = np.exp(2.0j*np.pi*zr*(kpt[2]+0.5))


        vect_int[ik] = np.einsum('i,j,k,ijk', ex1, ex2,ex3, Ψ_r) / (nk1 * nk2 * nk3)

    return vect_int

