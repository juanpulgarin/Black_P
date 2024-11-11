import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
import h5py

from band_tools import GetEshiftGap, GetBandsVect, GetIntensity, PlotIntensity, PlotCDAD
import Incoherent_Photo_Emission as Incoherent
import constants as cst
from pes_intensity import CalcAtomicMatrixElements
from python_w90 import wann90_tb
from orbdata_Hqpgw_new import GetOrbData_tb
#from orbdata_blackp import GetOrbData_tb

#out_path = "./../../OUT/"

def parameters_numerics(direccion,Q_plots,nk,out_path):
    nkx=nk[0]
    nky=nk[1]
    nkz=nk[2]

    x1 = np.linspace(-0.5,0.5, nkx+1)[0:-1]
    x2 = np.linspace(-0.5,0.5, nky+1)[0:-1]
    x3 = np.linspace(-0.5,0.5, nkz+1)[0:-1]

    Σy      = np.zeros((3,len(Q_plots)))
    X0y     = np.zeros((3,len(Q_plots)))
    X_max_y = np.zeros((3,len(Q_plots)))
    Y_max_y = np.zeros((3,len(Q_plots)))

    Ay      = np.zeros((3,len(Q_plots)))
    Cy      = np.zeros((3,len(Q_plots)))

    energia_exciton = np.zeros((len(Q_plots)))
    for ii_q in range(len(Q_plots)):
        if direccion == 'kx':
            QQ = np.array([Q_plots[ii_q],0.0,0.0])
        if direccion == 'ky':
            QQ = np.array([0.0,Q_plots[ii_q],0.0])
        if direccion == 'kz':
            QQ = np.array([0.0,0.0,Q_plots[ii_q]])

        """if np.sign(Q_plots[ii_q]) <  0.0:
            path_excitons = out_path  + "/out_Hqpgw_minus_new_merlin_new_screening"
        if np.sign(Q_plots[ii_q]) >  0.0:
            path_excitons = out_path  + "/out_Hqpgw_plus_new_merlin_new_screening"
        if np.sign(Q_plots[ii_q]) == 0.0:
            path_excitons = out_path  + "/out_Hqpgw_new_new_screening"""

        path_excitons = out_path

        ϵ_plus,ψ_plus = Incoherent.Reading_Data(QQ,path_excitons,retorno=5)
        pos_fix = np.where( np.reshape(ψ_plus[0,:], [nkx,nky,nkz]) ==np.max (np.reshape(ψ_plus[0,:], [nkx,nky,nkz])) )

        poptx_plus,popty_plus,poptz_plus = Incoherent.Fitting_Parameters(x1,nkx,nky,nkz,ψ_plus,direccion,-Q_plots[ii_q],pos_fix)

        Σy[:,ii_q] =  abs(np.array([poptx_plus[1],popty_plus[1],poptz_plus[1]]))
        X0y[:,ii_q] =  np.array([poptx_plus[2],popty_plus[2],poptz_plus[2]])

        xmax_p,ymax_p = Incoherent.maximos(x1,np.reshape(ψ_plus[0,:] , [nkx,nky,nkz])[:,pos_fix[1][0],pos_fix[2][0]])
        xmay_p,ymay_p = Incoherent.maximos(x2,np.reshape(ψ_plus[0,:] , [nkx,nky,nkz])[pos_fix[0][0],:,pos_fix[2][0]])
        xmaz_p,ymaz_p = Incoherent.maximos(x3,np.reshape(ψ_plus[0,:] , [nkx,nky,nkz])[pos_fix[0][0],pos_fix[1][0],:])

        X_max_y[:,ii_q] =  np.array([xmax_p,xmay_p,xmaz_p])
        Y_max_y[:,ii_q] =  np.array([ymax_p,ymay_p,ymaz_p])

        energia_exciton[ii_q] = ϵ_plus[0]


    Ay = ( -X0y  + X_max_y) / ( X0y * X_max_y - X_max_y**2 +  Σy )
    Cy =  Y_max_y/( ( 1+Ay * X_max_y ) * np.exp( -(X_max_y - X0y )**2/(2*Σy**2) ) )

    return Cy,Σy,X0y,Ay,X_max_y,Y_max_y,energia_exciton

def inter(Q_plots,data):
    #return spline(Q_plots, data[0,:])(Q_interp), spline(Q_plots, data[1,:])(Q_interp), spline(Q_plots, data[2,:])(Q_interp)
    return interpolate.interp1d(Q_plots, data[0,:],fill_value="extrapolate"), interpolate.interp1d(Q_plots, data[1,:],fill_value="extrapolate"), interpolate.interp1d(Q_plots, data[2,:],fill_value="extrapolate")


def AA_CC(X0x,X_max_x,Σx,Y_max_x,X0y,X_max_y,Σy,Y_max_y,X0z,X_max_z,Σz,Y_max_z):
    Ax = ( -X0x  + X_max_x) / ( X0x * X_max_x - X_max_x**2 +  Σx )
    Ay = ( -X0y  + X_max_y) / ( X0y * X_max_y - X_max_y**2 +  Σy )
    Az = ( -X0z  + X_max_z) / ( X0z * X_max_z - X_max_z**2 +  Σz )
    Cx =  Y_max_x/( ( 1+Ax * X_max_x ) * np.exp( -(X_max_x - X0x )**2/(2*Σx**2) ) )
    Cy =  Y_max_y/( ( 1+Ay * X_max_y ) * np.exp( -(X_max_y - X0y )**2/(2*Σy**2) ) )
    Cz =  Y_max_z/( ( 1+Az * X_max_z ) * np.exp( -(X_max_z - X0z )**2/(2*Σz**2) ) )

    return Ax,Ay,Az,Cx,Cy,Cz

def parameters(x1,QQ,Q_plots,X_max_x,Y_max_x,X0_x,Σ_x,X_max_y,Y_max_y,X0_y,Σ_y,X_max_z,Y_max_z,X0_z,Σ_z,ϵ_parameters):
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')

    Xmax_xi,Xmax_xj,Xmax_xk       = X_max_x[0](QQ[0]),X_max_x[1](QQ[0]),X_max_x[2](QQ[0])
    Ymax_xi,Ymax_xj,Ymax_xk       = Y_max_x[0](QQ[0]),Y_max_x[1](QQ[0]),Y_max_x[2](QQ[0])
    X0_xi,X0_xj,X0_xk             =    X0_x[0](QQ[0]),   X0_x[1](QQ[0]),   X0_x[2](QQ[0])
    Σ_xi,Σ_xj,Σ_xk                =     Σ_x[0](QQ[0]),    Σ_x[1](QQ[0]),    Σ_x[2](QQ[0])
    A_xi,A_xj,A_xk,C_xi,C_xj,C_xk = AA_CC(X0_xi,Xmax_xi,Σ_xi,Ymax_xi,X0_xj,Xmax_xj,Σ_xj,Ymax_xj,X0_xk,Xmax_xk,Σ_xk,Ymax_xk)

    Xmax_yi,Xmax_yj,Xmax_yk       = X_max_y[0](QQ[1]),X_max_y[1](QQ[1]),X_max_y[2](QQ[1])
    Ymax_yi,Ymax_yj,Ymax_yk       = Y_max_y[0](QQ[1]),Y_max_y[1](QQ[1]),Y_max_y[2](QQ[1])
    X0_yi,X0_yj,X0_yk             =    X0_y[0](QQ[1]),   X0_y[1](QQ[1]),   X0_y[2](QQ[1])
    Σ_yi,Σ_yj,Σ_yk                =     Σ_y[0](QQ[1]),    Σ_y[1](QQ[1]),    Σ_y[2](QQ[1])
    A_yi,A_yj,A_yk,C_yi,C_yj,C_yk = AA_CC(X0_yi,Xmax_yi,Σ_yi,Ymax_yi,X0_yj,Xmax_yj,Σ_yj,Ymax_yj,X0_yk,Xmax_yk,Σ_yk,Ymax_yk)

    Xmax_zi,Xmax_zj,Xmax_zk       = X_max_z[0](QQ[2]),X_max_z[1](QQ[2]),X_max_z[2](QQ[2])
    Ymax_zi,Ymax_zj,Ymax_zk       = Y_max_z[0](QQ[2]),Y_max_z[1](QQ[2]),Y_max_z[2](QQ[2])
    X0_zi,X0_zj,X0_zk             =    X0_z[0](QQ[2]),   X0_z[1](QQ[2]),   X0_z[2](QQ[2])
    Σ_zi,Σ_zj,Σ_zk                =     Σ_z[0](QQ[2]),    Σ_z[1](QQ[2]),    Σ_z[2](QQ[2])
    A_zi,A_zj,A_zk,C_zi,C_zj,C_zk = AA_CC(X0_zi,Xmax_zi,Σ_zi,Ymax_zi,X0_zj,Xmax_zj,Σ_zj,Ymax_zj,X0_zk,Xmax_zk,Σ_zk,Ymax_zk)


    ρ_total =  C_xk*( 1+ A_xk*Z) * ( 1+ A_yj*Y) * ( 1+ A_zi*X) \
        *np.exp( -(Z-X0_xk)**2/(2*Σ_xk**2) )*np.exp( -(Y-X0_yj)**2/(2*Σ_yj**2) )*np.exp( -(X-X0_zi)**2/(2*Σ_zi**2) )



    ϵ_total = ϵ_parameters[0] + ϵ_parameters[1]*QQ[0]**2 + ϵ_parameters[2]*QQ[1]**2 + ϵ_parameters[3]*QQ[2]**2

    return ρ_total,ϵ_total

def func_bands_q(x, a, c, x0):
    return  a *(x-x0)**2 + c

def energy_interp(Q_plots,ϵ_x,ϵ_y,ϵ_z):
    poptx, pcovx = curve_fit( func_bands_q,Q_plots,ϵ_x )
    popty, pcovy = curve_fit( func_bands_q,Q_plots,ϵ_y )
    poptz, pcovz = curve_fit( func_bands_q,Q_plots,ϵ_z )
    return poptx[1],poptx[0],popty[0],poptz[0]

def GenMatrixElements(tipo_Hamiltoniano,vectores,ws,kpath,polarization,bands,bandas_info,photon,project_band=-1,fout=''):
    a1 = vectores['a1']
    a3 = vectores['a2']
    a2 = vectores['a3']

    b1 = vectores['b1']
    b2 = vectores['b2']
    b3 = vectores['b3']

    wphot = photon['wphot'] / cst.Ry
    alpha = photon['alpha']/180. * np.pi
    Vin = photon['Vin'] / cst.Ry
    Phiw = photon['Phiw'] / cst.Ry
    Eshift = photon['Eshift'] / cst.Ry
    rint = photon['rint']
    eta = photon['eta']

    pol_tag = polarization['tag']
    pol = polarization['vector']

    kdir = kpath['dir']
    kpts = np.zeros([kpath['nk'],2])
    if kdir == 'x':

        kpts[:,0] = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])*b1[0]
        kpoints = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])*b1[0]
    if kdir == 'y':

        kpts[:,1] = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])*b2[1]
        kpoints = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])*b2[1]
    if kdir != 'y' and  kdir != 'x':
        angulo = kpath['angulo']
        #print(np.cos(angulo),np.sin(angulo))
        kpts[:,0] = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])*np.cos(angulo)*b1[0]
        kpts[:,1] = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])*np.sin(angulo)*b2[1]

        kpoints = np.linspace(-np.sqrt((kpath['kmin']*np.cos(angulo))**2+(kpath['kmin']*np.sin(angulo))**2), np.sqrt((kpath['kmax']*np.cos(angulo))**2+(kpath['kmax']*np.sin(angulo))**2), kpath['nk'])

    orb = GetOrbData_tb(rint[0], rint[1], rint[2])

    ws_shift = ws - Phiw
    mel_at = CalcAtomicMatrixElements(orb,ws_shift,wphot,Vin,eta,kpts)
    print(angulo,angulo*180/np.pi)
    #--------------------------------------------------------------------------------------
    system="BlackP"
    path = "../../DATA/data_excitons/"+system+"/"
    nkx=72
    nky=72
    nkz=72
    fname = path + tipo_Hamiltoniano + ".h5"
    #fname = path + "Hqpgw.h5"
    #fname = "./data/blackp.h5"

    #--------------------------------------------------------------------------------------

    ham = wann90_tb(fname, from_hdf5=True)

    nbnd = len(bands)
    norb = ham.num_wann
    nk = kpts.shape[0]
    kpts_red = np.zeros([nk, 3])
    epsk = np.zeros([nk,nbnd])
    rotk = np.zeros([nk,norb,nbnd], dtype=np.complex_)

    for i, kpt in enumerate(kpts):
        kpt3 = np.array([kpt[0], kpt[1], 0.])
        kpts_red[i,:] = ham.get_kreduced(kpt3)
        eps, evect = ham.get_eig(kpts_red[i,:])
        eps[bandas_info['scissor_index']:-1]+= bandas_info['scissor_energy']
        epsk[i, :] = eps[bands]
        rotk[i, :, :] = evect[:, bands]

    mel = np.einsum('cjkw, kja -> cakw', mel_at, rotk)
    mel_pol = np.einsum('cakw, c -> akw', mel, pol)

    matel_re = np.einsum('akw -> kwa', np.real(mel_pol) )
    matel_im = np.einsum('akw -> kwa', np.imag(mel_pol) )

    file_matel = f"data/matel_k{kdir}_pol{pol_tag}.h5"

    if project_band > -1:
        for a in range(nbnd):
            if bands[a] != project_band:
                matel_re[:,:,a] = 0.
                matel_im[:,:,a] = 0.

        file_matel = f"data/matel_proj{project_band}_k{kdir}_pol{pol_tag}.h5"

    f = h5py.File(file_matel, "w")
    f.attrs['nst'] = len(bands)
    f.attrs['nk'] = nk
    f.attrs['ne'] = len(ws)
    f.attrs['emin'] = ws[0]
    f.attrs['emax'] = ws[-1]
    f.create_dataset('real', data=matel_re)
    f.create_dataset('imag', data=matel_im)
    f.close()

    inten_p = np.zeros([kpath['nk'], len(ws)])
    for i, en in enumerate(ws):

        ##inten_p[:,i] = GetIntensity(epsk,np.ones_like(rotk),en,np.ones_like(mel_at[:,:,:,i]),pol,eta,1.5,Eshift=Eshift,band_range=[0,1],beta=200)
        inten_p[:,i] = GetIntensity(epsk,rotk,en,mel_at[:,:,:,i],pol,eta,1.5,Eshift=Eshift,band_range=[0,1],beta=200)
        #inten_p[:,i] = GetIntensity(epsk,np.ones_like(rotk),en,mel_at[:,:,:,i],pol,eta,0.0,Eshift=Eshift,band_range=[0,1],beta=200)
        #inten_p[:,i] = GetIntensity(epsk,rotk,en,np.ones_like(mel_at[:,:,:,i]),pol,eta,0.0,Eshift=Eshift,band_range=[0,1],beta=200)

    PlotIntensity(kpoints,ws,inten_p,angulo,fac=1.0,units='au',fout=fout)

    return mel_pol

