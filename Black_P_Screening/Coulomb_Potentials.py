import numpy as np
from scipy.optimize import curve_fit
# Import the Symmetric Fourier Transform class
from hankel import SymmetricFourierTransform
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import interpolate
import matplotlib.pyplot as plt

ft = SymmetricFourierTransform(ndim=3, N = 10000, h = .1)


k = np.logspace(-2.0,2,5000)
r = np.linspace(0.1,250,15000)
condicion_short_range = (np.array(r)<=18)
condicion_long_range = (np.array(r)>=18) & (np.array(r)<=35)
r_small = r[condicion_long_range]
delta_r = r_small[1]-r_small[0]

def ModelDielectric(q,eq0,n_bar,alpha):
    eps1 = 1 + 1 / (  1/(eq0 - 1) +alpha*(q/ (2*(3*n_bar/np.pi)**(1/6))   )**2 + q**4/(4*np.sqrt(4*np.pi*n_bar))  )

    return eps1

def funcH(x, cnm, rnm):
    return cnm/(x+rnm)

def funcV(x, cnm,rnm,anm):
    return cnm /(anm*x+rnm)

def rotation(control,ds,Ix,distancia):
    condicion_short_range = np.max(np.where(ds[Ix]<=distancia))
    if control==1:
        return (  np.arctan(1*(ds-ds[Ix][condicion_short_range]))  +np.pi/2)/np.max(np.arctan(1*(ds-ds[Ix][condicion_short_range])) +np.pi/2)
    if control==2:
        return (  np.arctan(-1*(ds-ds[Ix][condicion_short_range]))  +np.pi/2)/np.max(np.arctan(-1*(ds-ds[Ix][condicion_short_range])) +np.pi/2)

def Fourier_Data():
    y = np.array([-0.001,0.0,0.25,0.5,0.75,1.0])

    vr_I       = lambda x : 1/abs( x+y[0] )
    vr0        = lambda x : 1/abs( x+y[1] )
    vrII       = lambda x : 1/abs( x+y[2] )
    vrIIII     = lambda x : 1/abs( x+y[3] )
    vrIIIIII   = lambda x : 1/abs( x+y[4] )
    vrIIIIIIII = lambda x : 1/abs( x+y[5] )


    vk_I       = ft.transform( vr_I      , k, ret_err=False)
    vk0        = ft.transform( vr0       , k, ret_err=False)
    vkII       = ft.transform( vrII      , k, ret_err=False)
    vkIIII     = ft.transform( vrIIII    , k, ret_err=False)
    vkIIIIII   = ft.transform( vrIIIIII  , k, ret_err=False)
    vkIIIIIIII = ft.transform( vrIIIIIIII, k, ret_err=False)

    z=np.zeros((len(k),len(y)))

    z[:,0] = vk_I
    z[:,1] = vk0
    z[:,2] = vkII
    z[:,3] = vkIIII
    z[:,4] = vkIIIIII
    z[:,5] = vkIIIIIIII


    f = interpolate.interp2d(y,k,z, kind = 'cubic')

    return f

def Minimization_Alpha(irvec_v_bare,Vint_rT,irvec_W,Wint_rF,orbitales,lat,eq0,n_bar,alpha_array):
    a1, a2, a3 = lat[0,:], lat[1,:], lat[2,:]
    nwan = Vint_rT.shape[0]
    theta0 = [eq0,n_bar,0.0]

    big_error = np.zeros((len(alpha_array)))
    alpha_contador = 0

    f = Fourier_Data()

    print("interpolado")

    for i_alpha in alpha_array:

        theta0[2] = i_alpha

        epsilon = ModelDielectric(k,*theta0)

        error_array = np.zeros((int(nwan*(nwan+1)/2)))
        contador=0
        for i_orb in range(0,nwan):
            for j_orb in range(i_orb,nwan):
                if i_orb == j_orb:

                    xsT = irvec_v_bare[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
                    ysT = irvec_v_bare[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
                    zsT = irvec_v_bare[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

                    xsF = irvec_W[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
                    ysF = irvec_W[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
                    zsF = irvec_W[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

                    rvecsT = xsT[:,None] * a1[None,:] + ysT[:,None] * a2[None,:]+ zsT[:,None] * a3[None,:]
                    rvecsF = xsF[:,None] * a1[None,:] + ysF[:,None] * a2[None,:]+ zsF[:,None] * a3[None,:]

                    dsT = np.sqrt(rvecsT[:,0]**2 + rvecsT[:,1]**2 + rvecsT[:,2]**2)
                    dsF = np.sqrt(rvecsF[:,0]**2 + rvecsF[:,1]**2 + rvecsF[:,2]**2)

                    IxT = np.argsort(dsT)
                    IxF = np.argsort(dsF)

                    poptV, pcovV = curve_fit(funcH, dsT[IxT][1:], Vint_rT[i_orb,j_orb,IxT][1:])

                    vk = f(poptV[1],k)[:,0]

                    wq_sp = spline(k, vk/(epsilon))

                    wr = ft.transform(wq_sp,r, ret_err=False,  inverse=True) * poptV[0]

                    ymn = np.interp(r,dsF[IxF], Wint_rF[i_orb,j_orb,IxF])

                    wr_small  = wr[condicion_long_range]
                    ymn_small = ymn[condicion_long_range]

                    error = 0.0

                    for r_i in range(len(r_small)):
                        error+= abs(ymn_small[r_i]-wr_small[r_i])**2

                    error_array[contador] = error
                    contador+=1

        big_error[alpha_contador] = sum(error_array)/delta_r
        alpha_contador+=1

    return big_error

def Screening_Long_Interaction(irvec_v_bare,Vint_rT,irvec_W,Wint_rF,orbitales,lat,eq0,n_bar,alpha):
    W_epsilon = np.zeros_like(Wint_rF)

    a1, a2, a3 = lat[0,:], lat[1,:], lat[2,:]
    nwan = Vint_rT.shape[0]
    theta0 = [eq0,n_bar,alpha]

    epsilon = ModelDielectric(k,*theta0)

    f = Fourier_Data()

    for i_orb in range(0,nwan):
        for j_orb in range(0,nwan):

            xsT = irvec_v_bare[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
            ysT = irvec_v_bare[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
            zsT = irvec_v_bare[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

            xsF = irvec_W[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
            ysF = irvec_W[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
            zsF = irvec_W[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

            rvecsT = xsT[:,None] * a1[None,:] + ysT[:,None] * a2[None,:]+ zsT[:,None] * a3[None,:]
            rvecsF = xsF[:,None] * a1[None,:] + ysF[:,None] * a2[None,:]+ zsF[:,None] * a3[None,:]

            dsT = np.sqrt(rvecsT[:,0]**2 + rvecsT[:,1]**2 + rvecsT[:,2]**2)
            dsF = np.sqrt(rvecsF[:,0]**2 + rvecsF[:,1]**2 + rvecsF[:,2]**2)

            IxT = np.argsort(dsT)
            IxF = np.argsort(dsF)

            poptV, pcovV = curve_fit(funcH, dsT[IxT][1:], Vint_rT[i_orb,j_orb,IxT][1:])

            vk = f(poptV[1],k)[:,0]

            wq_sp = spline(k, vk/(epsilon))

            wr = ft.transform(wq_sp,r, ret_err=False,  inverse=True) * poptV[0]

            ymn = spline(r, wr)

            W_epsilon[i_orb,j_orb,IxF] = ymn(dsF[IxF])
        print(i_orb)
    return W_epsilon

def Screening_Short_Interaction(irvec_W,Wint_rF,orbitales,lat):
    W_epsilon_short = np.zeros_like(Wint_rF)

    a1, a2, a3 = lat[0,:], lat[1,:], lat[2,:]
    nwan = Wint_rF.shape[0]

    for i_orb in range(0,nwan):
        for j_orb in range(0,nwan):

                xsF = irvec_w_e_F_32_12[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
                ysF = irvec_w_e_F_32_12[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
                zsF = irvec_w_e_F_32_12[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

                rvecsF = xsF[:,None] * a1[None,:] + ysF[:,None] * a2[None,:]+ zsF[:,None] * a3[None,:]

                dsF = np.sqrt(rvecsF[:,0]**2 + rvecsF[:,1]**2 + rvecsF[:,2]**2)

                IxF = np.argsort(dsF)

                if i_orb != j_orb:
                    condicion_w = dsF[IxF]<=45
                    poptV, pcovV = curve_fit(funcV, dsF[IxF][condicion_w],Wint_rF[i_orb,j_orb,IxF][condicion_w])

                if i_orb == j_orb:
                    condicion_w = dsF[IxF[1:]]<=25
                    poptV, pcovV = curve_fit(funcV, dsF[IxF[1:]][condicion_w],Wint_rF[i_orb,j_orb,IxF[1:]][condicion_w])

                wr = funcV(dsF[IxF],*poptV)



                W_epsilon_short[i_orb,j_orb,IxF] = wr

                if i_orb == j_orb:
                    W_epsilon_short[i_orb,j_orb,IxF[0]] = Wint_rF[i_orb,j_orb,IxF[0]]


    return W_epsilon_short



###------------------------------------------------------------------------------------------------------------
"""def func_comparison(theta):
    xs = Fourier.Transform_r(12,12,8,ScreenVintq(nk1,nk2,nk3,Vint_q,lat,*theta) )[i_orb,j_orb,:]
    x=np.linspace(0,len(xs)+1,len(xs))

    interp_func = interp1d(x, xs, kind='linear', fill_value='extrapolate')

    interpolated_x = np.linspace(0,len(xs)+1,len(ys))
    interpolated_y = interp_func(interpolated_x)
    return interpolated_y-ys"""


###------------------------------------------------------------------------------------------------------------
"""eigenvalues_todos = np.zeros((len(dsT),32),dtype=np.complex_)
eigenvectors_todos = np.zeros((len(dsT),32,32),dtype=np.complex_)


original = np.zeros_like(Vint_rT[:,:,:],dtype=np.complex_)
modified = np.zeros_like(Vint_rT[:,:,:],dtype=np.complex_)


for i_ds in range(0,len(dsT)):


    eigenvalues, eigenvectors = LA.eig(Vint_rT[:,:,i_ds])
    eigenvalues_todos[i_ds,:] = eigenvalues
    eigenvectors_todos[i_ds,:,:] = eigenvectors

copied_eigenvalues_todos = eigenvalues_todos.copy()

popt0, pcov0 = curve_fit(func, dsT[IxT][1:], eigenvalues_todos[IxT,0][1:])

xsT = irvec_v_trial[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
ysT = irvec_v_trial[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
zsT = irvec_v_trial[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

rvecsT = xsT[:,None] * a1[None,:] + ysT[:,None] * a2[None,:]+ zsT[:,None] * a3[None,:]
dsT = np.sqrt(rvecsT[:,0]**2 + rvecsT[:,1]**2 + rvecsT[:,2]**2)


IxT = np.argsort(dsT)

poptV, pcovV = curve_fit(func, dsT[IxT][1:], eigenvalues_todos[IxT,0][1:])

vk0 = Fourier_Data()f(poptV[1],k)*poptV[0]

wq_sp0 = spline(k, vk0/(epsilon))

wr0 = ft.transform(wq_sp0,r, ret_err=False,  inverse=True)



w_larger_fit = interpolate.interp1d(r, wr0, fill_value='extrapolate')

copied_eigenvalues_todos[:,0] = w_larger_fit(dsT[:])


for i_ds in range(0,len(dsT)):

    original[:,:,i_ds] = np.dot( np.dot( eigenvectors_todos[i_ds,:,:],np.diag(eigenvalues_todos[i_ds,:]) ),  np.linalg.inv( eigenvectors_todos[i_ds,:,:]  )        )
    modified[:,:,i_ds] = np.dot( np.dot( eigenvectors_todos[i_ds,:,:],np.diag(copied_eigenvalues_todos[i_ds,:]) ),  np.linalg.inv( eigenvectors_todos[i_ds,:,:]  )        )



"""
