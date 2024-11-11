import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import Reading_Scripts as Reading


def func_q_nq_0(x, c, σ, x0,a):
    return c*(1.0 + a*x) * np.exp(-(x-x0)**2/(2*σ**2) )

def func_q_eq_0(x, c, σ, x0):
    return  c          * np.exp(-(x-x0)**2/(2*σ**2) )

def func_bands_q(x, c, x0):
    return  c *(x-x0)**2

def interpolador(direccion):
    if direccion == 'kz':
        fx = func_q_nq_0
        fy = func_q_eq_0
        fz = func_q_eq_0
        ft = func_q_nq_0

    if direccion == 'kx':
        fx = func_q_eq_0
        fy = func_q_eq_0
        fz = func_q_nq_0
        ft = func_q_nq_0

    if direccion == 'ky':
        fx = func_q_eq_0
        fy = func_q_nq_0
        fz = func_q_eq_0
        ft = func_q_nq_0

    return fx,fy,fz,ft

def maximos(x1,y_exciton):
    max_position = np.where( y_exciton == np.max( y_exciton )   )[0][0]

    x_max = x1[max_position]
    y_max = y_exciton[max_position]

    return x_max,y_max

def Reading_Data(QQ,path,retorno=5):
    nombre=[]
    for qq in range(len(QQ)):
        a = str(QQ[qq])
        if a=='0.0':
            a='0.00'
        nombre.append(a)

    fname = path+"/"+'px'+nombre[0]+'_py'+nombre[1]+'_pz'+nombre[2]+"_exc.h5"
    ϵ_plus,ϕ_real_plus,ϕ_imag_plus =Reading.Read_Eigen_states(fname)
    ϕ_plus = ϕ_real_plus + 1.0j*ϕ_imag_plus

    fname = path+"/"+'px'+nombre[0]+'_py'+nombre[1]+'_pz'+nombre[2]+"_exc_dens.h5"
    ψ_plus = Reading.Read_Density(fname)

    if retorno ==1:
        return ϵ_plus

    if retorno ==2:
        return ϕ_plus

    if retorno ==3:
        return ψ_plus

    if retorno ==4:
        return ϵ_plus,ϕ_plus

    if retorno ==5:
        return ϵ_plus,ψ_plus

    if retorno ==6:
        return ϕ_plus,ψ_plus

    if retorno ==7:
        return ϵ_plus,ϕ_plus,ψ_plus

def partition_weight(EQQ,β):
    return np.exp(-β*EQQ)

def nfermi(beta,w):
   return 1.0 / (1.0 + np.exp(beta*w))

"""def Reading_Data(Q,direccion,retorno=5):
    zero = '0.00'
    Dp =  np.array([Q])
    Dm = -Dp

    if retorno==8:
        fname="./out_Hqpgw_new/"+'px'+zero+'_py'+zero+'_pz'+zero+"_exc.h5"
        ϵ_plus,ϕ_real_plus,ϕ_imag_plus =Reading.Read_Eigen_states(fname)
        ϕ_plus = ϕ_real_plus + 1.0j*ϕ_imag_plus

        fname="./out_Hqpgw_new/"+'px'+zero+'_py'+zero+'_pz'+zero+"_exc_dens.h5"
        ψ_plus = Reading.Read_Density(fname)

    if direccion =='kx' and retorno !=8:
        fname="./out_Hqpgw_plus_new_merlin/"+'px'+str(Dp[0])+'_py'+zero+'_pz'+zero+"_exc.h5"
        ϵ_plus,ϕ_real_plus,ϕ_imag_plus =Reading.Read_Eigen_states(fname)
        ϕ_plus = ϕ_real_plus + 1.0j*ϕ_imag_plus

        fname="./out_Hqpgw_plus_new_merlin/"+'px'+str(Dp[0])+'_py'+zero+'_pz'+zero+"_exc_dens.h5"
        ψ_plus = Reading.Read_Density(fname)
        #--------------------------------------------------------------------------------------------
        fname="./out_Hqpgw_minus_new_merlin/"+'px'+str(Dm[0])+'_py'+zero+'_pz'+zero+"_exc.h5"
        ϵ_minus,ϕ_real_minus,ϕ_imag_minus =Reading.Read_Eigen_states(fname)
        ϕ_minus = ϕ_real_minus + 1.0j*ϕ_imag_minus

        fname="./out_Hqpgw_minus_new_merlin/"+'px'+str(Dm[0])+'_py'+zero+'_pz'+zero+"_exc_dens.h5"
        ψ_minus = Reading.Read_Density(fname)

    if direccion =='ky' and retorno !=8:
        fname="./out_Hqpgw_plus_new_merlin/"+'px'+zero+'_py'+str(Dp[0])+'_pz'+zero+"_exc.h5"
        #fname="./out_Hqpgw_ky_test/"+'px'+zero+'_py'+str(Dp[0])+'_pz'+zero+"_exc.h5"
        ϵ_plus,ϕ_real_plus,ϕ_imag_plus =Reading.Read_Eigen_states(fname)
        ϕ_plus = ϕ_real_plus + 1.0j*ϕ_imag_plus

        fname="./out_Hqpgw_plus_new_merlin/"+'px'+zero+'_py'+str(Dp[0])+'_pz'+zero+"_exc_dens.h5"
        #fname="./out_Hqpgw_ky_test/"+'px'+zero+'_py'+str(Dp[0])+'_pz'+zero+"_exc_dens.h5"
        ψ_plus = Reading.Read_Density(fname)
        #--------------------------------------------------------------------------------------------
        fname="./out_Hqpgw_minus_new_merlin/"+'px'+zero+'_py'+str(Dm[0])+'_pz'+zero+"_exc.h5"
        #fname="./out_Hqpgw_ky_test/"+'px'+zero+'_py'+str(Dm[0])+'_pz'+zero+"_exc.h5"
        ϵ_minus,ϕ_real_minus,ϕ_imag_minus =Reading.Read_Eigen_states(fname)
        ϕ_minus = ϕ_real_minus + 1.0j*ϕ_imag_minus

        fname="./out_Hqpgw_minus_new_merlin/"+'px'+zero+'_py'+str(Dm[0])+'_pz'+zero+"_exc_dens.h5"
        #fname="./out_Hqpgw_ky_test/"+'px'+zero+'_py'+str(Dm[0])+'_pz'+zero+"_exc_dens.h5"
        ψ_minus = Reading.Read_Density(fname)

    if direccion =='kz' and retorno !=8:
        fname="./out_Hqpgw_plus_new_merlin/"+'px'+zero+'_py'+zero+'_pz'+str(Dp[0])+"_exc.h5"
        ϵ_plus,ϕ_real_plus,ϕ_imag_plus =Reading.Read_Eigen_states(fname)
        ϕ_plus = ϕ_real_plus + 1.0j*ϕ_imag_plus

        fname="./out_Hqpgw_plus_new_merlin/"+'px'+zero+'_py'+zero+'_pz'+str(Dp[0])+"_exc_dens.h5"
        ψ_plus = Reading.Read_Density(fname)
        #--------------------------------------------------------------------------------------------
        fname="./out_Hqpgw_minus_new_merlin/"+'px'+zero+'_py'+zero+'_pz'+str(Dm[0])+"_exc.h5"
        ϵ_minus,ϕ_real_minus,ϕ_imag_minus =Reading.Read_Eigen_states(fname)
        ϕ_minus = ϕ_real_minus + 1.0j*ϕ_imag_minus

        fname="./out_Hqpgw_minus_new_merlin/"+'px'+zero+'_py'+zero+'_pz'+str(Dm[0])+"_exc_dens.h5"
        ψ_minus = Reading.Read_Density(fname)

    if retorno ==1:
        return ϵ_plus,ϵ_minus

    if retorno ==2:
        return ϕ_plus,ϕ_minus

    if retorno ==3:
        return ψ_plus,ψ_minus

    if retorno ==4:
        return ϵ_plus,ϕ_plus,ϵ_minus,ϕ_minus

    if retorno ==5:
        return ϵ_plus,ψ_plus,ϵ_minus,ψ_minus

    if retorno ==6:
        return ϕ_plus,ψ_plus,ϕ_minus,ψ_minus

    if retorno ==7:
        return ϵ_plus,ϕ_plus,ψ_plus,ϕ_minus,ψ_minus

    if retorno ==8:
        return ϵ_plus,ψ_plus"""

def Plotting_Exitonic_Density(x1,nkx,nky,nkz,ψ,Qd,direccion):
    #====================================================================================
    fig_xy, ax_xy = plt.subplots()
    fig_zy, ax_xz = plt.subplots()
    fig_yz, ax_yz = plt.subplots()
    #====================================================================================

    #====================================================================================
    xy = ax_xy.imshow(np.transpose(np.reshape(ψ[0,:], [nkx,nky,nkz])[:,:,int(nkz/2)]),\
                extent=[x1.min(),x1.max(),x1.min(),x1.max()],
            origin='lower',aspect='auto',interpolation ='spline36',cmap = 'PuBu')

    ax_xy.set_xlabel(r'$k_x$')
    ax_xy.set_ylabel(r'$k_y$')
    ax_xy.axvline(x=0,color='black',linestyle='--')
    ax_xy.axhline(y=0,color='black',linestyle='--')

    ax_xy.set_xlim(-0.1,0.1)
    ax_xy.set_ylim(-0.1,0.1)
    #====================================================================================

    #====================================================================================
    xz = ax_xz.imshow(np.transpose(np.reshape(ψ[0,:], [nkx,nky,nkz])[:,int(nky/2),:]),\
                extent=[x1.min(),x1.max(),x1.min(),x1.max()],
            origin='lower',aspect='auto',interpolation ='spline36',cmap = 'PuBu')

    ax_xz.set_xlabel(r'$k_x$')
    ax_xz.set_ylabel(r'$k_z$')
    ax_xz.axvline(x=0,color='black',linestyle='--')
    ax_xz.axhline(y=0,color='black',linestyle='--')

    ax_xz.set_xlim(-0.1,0.1)
    ax_xz.set_ylim(-0.1,0.1)
    #====================================================================================

    #====================================================================================
    yz = ax_yz.imshow(np.transpose(np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,:]),\
                extent=[x1.min(),x1.max(),x1.min(),x1.max()],
            origin='lower',aspect='auto',interpolation ='spline36',cmap = 'PuBu')

    ax_yz.set_xlabel(r'$k_y$')
    ax_yz.set_ylabel(r'$k_z$')
    ax_yz.axvline(x=0,color='black',linestyle='--')
    ax_yz.axhline(y=0,color='black',linestyle='--')
    ###ax_yz.axhline(y=Qd,color='black',linestyle=':')
    ###ax_yz.axhline(y=Qd/2,color='red',linestyle=':')
    ax_yz.set_xlim(-0.1,0.1)
    ax_yz.set_ylim(-0.1,0.1)
    #====================================================================================

    #====================================================================================
    fig_x, ax_x = plt.subplots()
    fig_y, ax_y = plt.subplots()
    fig_z, ax_z = plt.subplots()
    #====================================================================================

    #====================================================================================
    #plt.plot(x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],'.')
    ax_x.plot(x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,int(nky/2),int(nkz/2)],'.')
    ax_x.set_xlabel(r'$k_x$')
    ax_x.axvline(x=0.0,color='black',linestyle='--')
    ax_x.set_xlim(-0.1,0.1)
    #====================================================================================

    #====================================================================================
    #plt.plot(x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],'.')
    ax_y.plot(x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],'.')
    ax_y.set_xlabel(r'$k_y$')
    ax_y.axvline(x=0.0,color='black',linestyle='--')
    ax_y.set_xlim(-0.1,0.1)
    #====================================================================================

    #====================================================================================
    #plt.plot(x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],'.')
    ax_z.plot(x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),int(nky/2),:],'.')
    ax_z.set_xlabel(r'$k_z$')
    ax_z.axvline(x=0.0,color='black',linestyle='--')
    ax_z.set_xlim(-0.1,0.1)
    #====================================================================================

    if direccion =='kx':
        ax_xz.axhline(y=Qd,color='black',linestyle=':')
        ax_xz.axhline(y=Qd/2,color='red',linestyle=':')
        ax_yz.axhline(y=Qd,color='black',linestyle=':')
        ax_yz.axhline(y=Qd/2,color='red',linestyle=':')

        ax_xz.set_ylim(Qd-0.1,-Qd+0.1)
        ax_yz.set_ylim(Qd-0.1,-Qd+0.1)

        ax_z.axvline(x=Qd,color='black',linestyle=':')
        ax_z.axvline(x=Qd/2,color='red',linestyle=':')

        ax_z.set_xlim(Qd-0.1,-Qd+0.1)

    if direccion =='ky':
        ax_xy.axhline(y=-Qd,color='black',linestyle=':')
        ax_xy.axhline(y=-Qd/2,color='red',linestyle=':')
        ax_yz.axvline(x=-Qd,color='black',linestyle=':')
        ax_yz.axvline(x=-Qd/2,color='red',linestyle=':')

        ax_xy.set_ylim(Qd-0.1,-Qd+0.1)
        ax_yz.set_xlim(Qd-0.1,-Qd+0.1)

        ax_y.axvline(x=-Qd,color='black',linestyle=':')
        ax_y.axvline(x=-Qd/2,color='red',linestyle=':')

        ax_y.set_xlim(Qd-0.1,-Qd+0.1)

    if direccion =='kz':
        ax_xy.axvline(x=Qd,color='black',linestyle=':')
        ax_xy.axvline(x=Qd/2,color='red',linestyle=':')
        ax_xz.axvline(x=Qd,color='black',linestyle=':')
        ax_xz.axvline(x=Qd/2,color='red',linestyle=':')

        ax_xy.set_xlim(Qd-0.1,-Qd+0.1)
        ax_xz.set_xlim(Qd-0.1,-Qd+0.1)

        ax_x.axvline(x=Qd,color='black',linestyle=':')
        ax_x.axvline(x=Qd/2,color='red',linestyle=':')

        ax_x.set_xlim(Qd-0.1,-Qd+0.1)

    plt.show()

"""def Fitting_Parameters(x1,nkx,nky,nkz,ψ,direccion,QQ):
    if direccion =='kx':
        poptx, pcovx = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,int(nky/2),int(nkz/2)],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)) )
        popty, pcovy = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)) )
        poptz, pcovz = curve_fit( func_q_nq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),int(nky/2),:],p0=np.array([1,1,-QQ,-np.sign(QQ)]), bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf,np.inf)), method = 'trf' )
        poptx[2] = 0.0
        popty[2] = 0.0

    if direccion =='ky':
        poptx, pcovx = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,int(nky/2),int(nkz/2)],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)) )
        #popty, pcovy = curve_fit( func_q_nq_0, x1[int(nky/2)-10:int(nky/2)+10],np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)][int(nky/2)-10:int(nky/2)+10] )
        popty, pcovy = curve_fit( func_q_nq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],p0=np.array([1,1,-QQ,-np.sign(QQ)]), bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        poptz, pcovz = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),int(nky/2),:],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)) )
        poptx[2] = 0.0
        poptz[2] = 0.0

    if direccion =='kz':
        poptx, pcovx = curve_fit( func_q_nq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,int(nky/2),int(nkz/2)],p0=np.array([1,1,-QQ,-np.sign(QQ)]), bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf,np.inf)), method = 'trf' )
        popty, pcovy = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)))
        poptz, pcovz = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),int(nky/2),:],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)))
        popty[2] = 0.0
        poptz[2] = 0.0

    return poptx,popty,poptz"""

def Fitting_Parameters(x1,nkx,nky,nkz,ψ,direccion,QQ,pos_fix):
    #pos_fix = np.where(abs(x1-QQ/2)<=0.001)[0][0]
    if direccion =='kx':
        poptx, pcovx = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,pos_fix[1][0],pos_fix[2][0]],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        popty, pcovy = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[pos_fix[0][0],:,pos_fix[2][0]],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        poptz, pcovz = curve_fit( func_q_nq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[pos_fix[0][0],pos_fix[1][0],:],p0=np.array([1,1,-QQ,-np.sign(QQ)]), bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        poptx[2] = 0.0
        popty[2] = 0.0

    if direccion =='ky':
        poptx, pcovx = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,pos_fix[1][0],pos_fix[2][0]],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        #popty, pcovy = curve_fit( func_q_nq_0, x1[int(nky/2)-10:int(nky/2)+10],np.reshape(ψ[0,:], [nkx,nky,nkz])[int(nkx/2),:,int(nkz/2)][int(nky/2)-10:int(nky/2)+10] )
        popty, pcovy = curve_fit( func_q_nq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[pos_fix[0][0],:,pos_fix[2][0]],p0=np.array([1,1,-QQ,-np.sign(QQ)]), bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        poptz, pcovz = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[pos_fix[0][0],pos_fix[1][0],:],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)), method = 'trf', max_nfev=1000 )
        poptx[2] = 0.0
        poptz[2] = 0.0

    if direccion =='kz':
        poptx, pcovx = curve_fit( func_q_nq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[:,pos_fix[1][0],pos_fix[2][0]],p0=np.array([1,1,-QQ,-np.sign(QQ)]), bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf,np.inf)), method = 'trf', max_nfev=2000 )
        popty, pcovy = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[pos_fix[0][0],:,pos_fix[2][0]],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)), method = 'trf', max_nfev=2000 )
        poptz, pcovz = curve_fit( func_q_eq_0, x1,np.reshape(ψ[0,:], [nkx,nky,nkz])[pos_fix[0][0],pos_fix[1][0],:],p0=np.array([1,1,0]), bounds=((-np.inf,-np.inf,-np.inf), (np.inf,np.inf,np.inf)), method = 'trf', max_nfev=2000 )
        popty[2] = 0.0
        poptz[2] = 0.0

    return poptx,popty,poptz



