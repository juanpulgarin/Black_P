import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
# ---- local modules -----
import constants as cst
from colormaps import cmap_arpes3
#-------------------------------------------------------------------
def nfermi(beta,w):
   return 1.0 / (1.0 + np.exp(beta*w))
#----------------------------------------------------------------------
def GetEshiftGap(wann,band=6):
   eps, vect = wann.get_eig(np.array([1./3., 1./3.]))
   return eps[band],eps[band+1]-eps[band]
#----------------------------------------------------------------------
def GetBandsVect(wann,kvecs,kz,Eshift=0.0):
   nk = kvecs.shape[0]

   kpts = np.zeros([kvecs.shape[0],3])
   for i, kvec in enumerate(kvecs):
      kvec3d = np.array([kvec[0], kvec[1], kz])
      kpts[i,:] = np.linalg.solve(wann.recip_lattice.T, kvec3d)
      
   epsk, vectk = wann.get_eig(kpts)

   return epsk + Eshift, vectk
#----------------------------------------------------------------------
def GetIntensity(epsk,vectk,ebind,mel_at,pol,eta,mu,Eshift=0.0,band_range=[],beta=200.0):
   nk = epsk.shape[0]
   epsk_ = epsk + Eshift

   bands_ = np.arange(0, epsk.shape[1], step=1)
   if len(band_range) > 0:
      bands_ = band_range

   inten = np.zeros(nk)

   for i, ibnd in enumerate(bands_):
      occ = nfermi(beta, epsk_[:,ibnd] - mu)
      dlt = np.exp(-0.5*(epsk_[:,ibnd] - ebind)**2/eta**2) / np.sqrt(2.0*np.pi*eta**2)	
      Ik, = np.where((occ > 1.0e-4) & (dlt > 1.0e-4))
      if len(Ik) > 0:
         mel_pol = mel_at[0,:,Ik] * pol[0] + mel_at[1,:,Ik] * pol[1] + mel_at[2,:,Ik] * pol[2]
         mel_band = np.einsum('kj,kj->k',vectk[Ik,:,ibnd],mel_pol)
         inten[Ik] = inten[Ik] + occ[Ik] * dlt[Ik] * np.abs(mel_band)**2
         #inten[Ik] = inten[Ik] + occ[Ik] * dlt[Ik]

   return inten
#----------------------------------------------------------------------
def PlotIntensity(ks,ws,inten,angulo,fac=1.0,Imax='none',units='au',fout=''):
   aB = cst.aB

   mycmap = cmap_arpes3()

   fig, ax = plt.subplots()

   if Imax == 'none':
      Imax_ = np.amax(inten) / fac
   else:
      Imax_ = Imax

   if units == 'au':
      im = ax.imshow(inten.T / Imax_, origin="lower", extent=(ks[0],ks[-1],cst.Ry*ws[0],cst.Ry*ws[-1]),\
      vmin=0.0,vmax=1.0,cmap=mycmap,aspect=0.2,interpolation='bicubic')
      ax.text(0,(cst.Ry*ws[0]+cst.Ry*ws[-1])/2,r'$\theta = %.1f$'%( (angulo)*180/np.pi-90) )

      ax.set_xlabel(r"$k_\parallel \, $ (a.u.)")
      ax.set_ylabel(r"$E\, $ (eV)")
      plt.grid()
   else:         
      im = ax.imshow(inten.T / Imax_, origin="lower", extent=(ks[0]/aB,ks[-1]/aB,cst.Ry*ws[0],cst.Ry*ws[-1]),\
      vmin=0.0,vmax=1.0,cmap=mycmap,aspect=0.2,interpolation='bicubic')
      ax.set_xlabel(r"$k_\parallel \, $ (a.u.)")
      ax.set_ylabel(r"$E\, $ (eV)")
      
   cs = fig.colorbar(im)

   if len(fout) > 0:
      plt.savefig(fout, bbox_inches='tight')
   else:
      plt.show()   
#----------------------------------------------------------------------
def PlotCDAD(kx_min,kx_max,ky_min,ky_max,nk1,nk2,cdad,fac=1.0,Imax='none',units='au',fout=''):
   aB = 0.5291772

   fig, ax = plt.subplots()

   if Imax == 'none':
      Imax_ = np.amax(np.abs(cdad)) / fac
   else:
      Imax_ = Imax

   spect = np.reshape(cdad, [nk1,nk2])

   if units == 'au':
      alat = 3.32 / 0.5291772
      im = ax.imshow(spect.T / Imax_, origin="lower", extent=(kx_min,kx_max,ky_min,ky_max),\
      vmin=-1.0,vmax=1.0,cmap=cm.bwr,aspect=1.0,interpolation='bicubic')
      DrawBZ(alat,ax,rot_ang=0)
      ax.set_xlabel(r"$k_x$ (a.u.)")
      ax.set_ylabel(r"$k_y$ (a.u.)")
   else:         
      alat = 3.32
      im = ax.imshow(spect.T / Imax_, origin="lower", extent=(kx_min/aB,kx_max/aB,ky_min/aB,ky_max/aB),\
      vmin=-1.0,vmax=1.0,cmap=cm.bwr,aspect=1.0,interpolation='bicubic')
   DrawBZ(alat,ax,rot_ang=0)
   ax.set_xlabel(r"$k_x (\AA^{-1})$")
   ax.set_ylabel(r"$k_y (\AA^{-1})$")

   # ax.set_xlim(-1.75, 1.75)
   # ax.set_ylim(-1.75, 1.75)

   cs = fig.colorbar(im)

   if len(fout) > 0:
      plt.savefig(fout, bbox_inches='tight')
   else:
      plt.show()   
#----------------------------------------------------------------------
