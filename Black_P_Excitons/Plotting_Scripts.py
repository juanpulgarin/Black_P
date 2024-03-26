import matplotlib.pyplot as plt
import numpy as np
import constants as cst

def Plot_bandstructure(epsk,klabel,Emin=-100.0,Emax=100.0,fout=""):
    nk   = epsk.shape[0]
    nbnd = epsk.shape[1]
    xk   = np.linspace(0.0, 1.0, nk)

    fig, ax = plt.subplots()

    for ibnd in range(nbnd):
        #ax.plot(xk, Ry * epsk[:,ibnd], c='blue')
        ax.plot(xk, cst.Ry * epsk[:,ibnd], c='red')
    Erange = np.amax(epsk) - np.amin(epsk)
    Emin_ = Ry*(np.amin(epsk) - 0.02 * Erange)
    Emax_ = Ry*(np.amax(epsk) + 0.02 * Erange)

    if Emin > -90.0:
        Emin_ = Emin
    if Emax < 90.0:
        Emax_ = Emax

    ax.set_ylim(Emin_,Emax_)

    ax.set_xlim(0.0,1.0)
    ax.set_ylabel(r'$Energy$ (eV)')

    knode = np.linspace(0,1,len(klabel))
    for i in range(1,len(klabel)-1):
        ax.axvline(x=knode[i],c='k', ls='--')

    ax.set_xticks(knode)
    ax.set_xticklabels(klabel)

    if len(fout) > 0:
        plt.savefig(fout+".pdf", bbox_inches="tight")
    else:
        plt.show()
        
def PlotVintrold(irvec,Vint_r,orbitales,lat,i_orb,j_orb,color_l,label_l,fout=""):
    xs = irvec[:,0] + (orbitales[i_orb,0] - orbitales[j_orb,0])
    ys = irvec[:,1] + (orbitales[i_orb,1] - orbitales[j_orb,1])
    zs = irvec[:,2] + (orbitales[i_orb,2] - orbitales[j_orb,2])

    a1, a2, a3 = lat[0,:], lat[1,:], lat[2,:]

    rvecs = xs[:,None] * a1[None,:] + ys[:,None] * a2[None,:]+ zs[:,None] * a3[None,:]
    ds = np.sqrt(rvecs[:,0]**2 + rvecs[:,1]**2 + rvecs[:,2]**2)


    Ix = np.argsort(ds)

    fig, ax = plt.subplots()

    ax.plot(ds[Ix], cst.Ry * Vint_r[i_orb,j_orb,Ix],".",color=color_l,label=label_l)
    ax.legend(fontsize=13)
    ax.set_xlabel(r"Distance $(\AA)$",fontsize=13)
    ax.set_ylabel(r"$V_{bare}\left(\left| r \right|\right)\  (eV)$",fontsize=13)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    
    if len(fout) > 0:
        plt.savefig(fout+"_"+str(i_orb)+"_"+str(j_orb)+".pdf", bbox_inches="tight", transparent=True)
        plt.savefig(fout+"_"+str(i_orb)+"_"+str(j_orb)+".png", bbox_inches="tight", transparent=True)
    else:
        plt.show()

def PlotVintq(nk1,nk2,nk3,iorb,jorb,Vint_q):
    fig, ax = plt.subplots(8,8,sharex=True,sharey=True,figsize=(8,8))


    f = np.reshape(np.real(Vint_q[:,iorb,jorb]), [nk1,nk2,nk3])

    ax[jorb,iorb].imshow(f.T, origin="lower", extent=(0,1,0,1))

    plt.tight_layout()
    plt.show()

def PlotVintr(irvec,Vint_r,Uint_r,orbitales,lat,npoints,i_orb,j_orb,color_l="black"):
    xs = irvec[:,0] - (orbitales[i_orb,0] - orbitales[j_orb,0])
    ys = irvec[:,1] - (orbitales[i_orb,1] - orbitales[j_orb,1])
    zs = irvec[:,2] - (orbitales[i_orb,2] - orbitales[j_orb,2])

    a1, a2, a3 = lat[0,:], lat[1,:], lat[2,:]

    rvecs = xs[:,None] * a1[None,:] + ys[:,None] * a2[None,:]+ zs[:,None] * a3[None,:]
    ds = np.sqrt(rvecs[:,0]**2 + rvecs[:,1]**2 + rvecs[:,2]**2)


    Ix = np.argsort(ds)



    if i_orb == j_orb:
        Vint_r[np.where(abs(ds)==0)[0][0],i_orb,i_orb]=Uint_r[i_orb]

    fig, ax = plt.subplots()

    ax.plot(ds[Ix], cst.Ry * Vint_r[Ix,i_orb,j_orb],"--",color=color_l,label=npoints)
    ax.set_xlabel(r"Distance $(\AA)$",fontsize=13)
    ax.set_ylabel(r"$W^{Hubb}\left(\left| r \right|\right)\  (eV)$",fontsize=13)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    #ax.set_ylim(0,1)
    ax.legend()
