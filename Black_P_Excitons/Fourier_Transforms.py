import numpy as np
import scipy.fftpack as sfft

def Transform_q(nk1,nk2,nk3,Vint_r):
    nwan = Vint_r.shape[0]
    nR = Vint_r.shape[-1]

    Vint_q = np.zeros([nwan,nwan,nR], dtype=np.complex_)

    for i in range(nwan):
        for j in range(nwan):
            
            x = np.reshape(Vint_r[i,j,:], [nk1,nk2,nk3])
            y = sfft.ifftn(x)
            y_shift = sfft.fftshift(y)
            Vint_q[i,j,:] = np.reshape(y_shift, [nR])
            
    return Vint_q

def Transform_r(nk1,nk2,nk3,Vint_q,bandera=1):
    if bandera==1:
        nwan = Vint_q.shape[0]
        nR = Vint_q.shape[-1]

        Vint_r = np.zeros([nwan,nwan,nR])

        for i in range(nwan):
            for j in range(nwan):

                x = np.reshape(Vint_q[i,j,:], [nk1,nk2,nk3])
                x_shift = sfft.fftshift(x)
                y = sfft.fftn(x_shift)
                Vint_r[i,j,:] = np.reshape(np.real(y), [nR])
    if bandera ==2:
        nR = Vint_q.shape[-1]

        Vint_r = np.zeros([nR])

        x = np.reshape(Vint_q[:], [nk1,nk2,nk3])
        x_shift = sfft.fftshift(x)
        y = sfft.fftn(x_shift)
        Vint_r[:] = np.reshape(np.real(y), [nR])

    return Vint_r

