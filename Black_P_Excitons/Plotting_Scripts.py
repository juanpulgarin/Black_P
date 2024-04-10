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
    Emin_ = cst.Ry*(np.amin(epsk) - 0.02 * Erange)
    Emax_ = cst.Ry*(np.amax(epsk) + 0.02 * Erange)

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

###plotting 3d
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


x, y = np.linspace(0, nkx-1, nkx), np.linspace(0, nky-1, nky)
X,Y = np.meshgrid(x,y)
#fig = go.Figure(data=[go.Surface(z=ψ[0,:].reshape(nkx,nky,nkz)[:,:,16], x=x, y=y,
fig = go.Figure(data=[go.Surface(z=ψ[0,:].reshape(nkx,nky,nkz)[:,:,int(nkz/2)], x=x, y=y,
                        colorscale='Reds',cmin=0,cmax=0.6, colorbar = dict(tickfont=dict(
                            color='black',size=30,family='Old Standard TT, serif',),orientation='h')   )])
#fig = ff.create_trisurf(z=ψ[0,:].reshape(32,32,32)[:,:,16], x=X, y=Y)

fig.update_layout(scene = dict(
                    xaxis = dict( title = ' ',
                        ticktext= [r'-1/4',r'0',r'1/4'],
                        tickvals= [int(nkx/2)-int(nkx/4),int(nkx/2)-1,int(nkx/2)+int(nkx/4)-1],
                        range=[int(nkx/2)-int(nkx/4)-1,int(nkx/2)+int(nkx/4)],tickfont=dict(
                            color='black',
                            size=17,
                            family='Old Standard TT, serif',)),
                    yaxis = dict( title = ' ',
                        ticktext= [r'-1/4',r'0',r'1/4'],
                        tickvals= [int(nky/2)-int(nky/4),int(nky/2)-1,int(nky/2)+int(nky/4)-1],
                        range=[int(nky/2)-int(nky/4)-1,int(nky/2)+int(nky/4)],tickfont=dict(
                            color='black',
                            size=17,
                            family='Old Standard TT, serif',)) ,
                    zaxis = dict( title = ' ',
                            tickfont=dict(
                            color='black',
                            size=17,
                            family='Old Standard TT, serif',) )),
                    width=700,
                    margin=dict(r=10, l=10, b=10, t=10)
                  )


#fig.update_xaxes(range=[8, 23])
#fig.update_layout(xaxis = dict(range=[7,24]))
#fig.update_yaxes(range=[8, 23])


fig.show()

fig, ax = plt.subplots(1, 1)
palette = copy(plt.get_cmap('Reds'))
palette.set_under('white', 1.0)  # 1.0 represents not transparent
#levels = np.arange(0.0,0.6, 0.01)
#levels[0] = 1e-3
#norm = colors.BoundaryNorm(levels, ncolors=palette.N)


#contour = ax.imshow(ψ[0,:].reshape(32,32,32)[:,:,16],aspect='auto', cmap=palette,norm=norm)
contour = ax.contourf(ψ[0,:].reshape(nkx,nky,nkz)[:,:,int(nkz/2)], cmap=palette)
#cbar = fig.colorbar(contour, extend='min', shrink=0.9, ax=ax)

"""
