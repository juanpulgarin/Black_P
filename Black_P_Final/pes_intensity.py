import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pickle
import time
# ---- local modules -----
import constants as cst
import tools
from orbdata_Hqpgw_new import GetOrbData_tb
from band_tools import GetEshiftGap, GetBandsVect, GetIntensity, \
   PlotIntensity, PlotCDAD
from geometry import LightGeometry
# ---- external modules -----
import arpes
from python_w90 import wann90_tb
#----------------------------------------------------------------------
def ReadCoeffs(fname):
   f = open(fname, "rb")
   data = pickle.load(f)
   f.close()

   return data['eb'], data['rint'], \
      data['vin'], data['phase']
#----------------------------------------------------------------------
def CalcAtomicMatrixElements(orb,ebind,wphot,Vin,eta,kpts):
   nk = kpts.shape[0]
   nw = len(ebind)

   mel_at = np.zeros([3,orb['norb'],nk,nw],dtype=np.complex_)
   for i, en in enumerate(ebind):
      mel_at[:,:,:,i] = arpes.CalcAtomicMatrixElements_length(kpts,en,wphot,eta,orb,\
                                                              Vinner=Vin,mbasis=False)

   return mel_at
#----------------------------------------------------------------------
def main(argv):

   Phiw = 4.95409
   #Evac = 0.0
   wphot = 6.2 / cst.Ry
   Eshift = (-6.594 - Phiw) / cst.Ry
   iscis = 20
   Escis = 0.27 / cst.Ry
   eta = 0.002

   alpha = 54 / 180 * np.pi
   geo = LightGeometry(alpha, 1.0)

   file_ham = 'data/blackp.h5'
   wann = wann90_tb(file_ham, from_hdf5=True)

   b1 = wann.recip_lattice[0,0:2]
   b2 = wann.recip_lattice[1,0:2]

   Vin = 0.
   #mu = Evac + 0.025/ cst.Ry + 5.0/cst.Ry
   mu = (Phiw + 0.025) / cst.Ry + 5.0/cst.Ry
   rint = [0.2, 1.0, 2.0]

   ktag = "x"
   file_kpts = "data/path_{}_n64.dat".format(ktag)
   kpts = np.loadtxt(file_kpts)
   nk = kpts.shape[0]

   #Emin, Emax = -Evac - 0.5 / cst.Ry, -Evac + 1.5 / cst.Ry
   Emin, Emax = -Phiw/cst.Ry - 0.5 / cst.Ry, -Phiw/cst.Ry + 1.5 / cst.Ry
   ebind = np.linspace(Emin, Emax, 51)
 
   tic = time.perf_counter()
   orb = GetOrbData_tb(rint[0], rint[1], rint[2])

   mel_at = CalcAtomicMatrixElements(orb,ebind,wphot,Vin,eta,kpts)
   toc = time.perf_counter()
   tools.PrintTime(tic,toc,"atomic matrix elements")

   tic = time.perf_counter()

   kz = 0.
   epsk, vectk = GetBandsVect(wann,kpts,kz,Eshift=Eshift)
   epsk[:,iscis:] += Escis

   '''
   fig, ax = plt.subplots()
   for i in range(epsk.shape[1]):
      ax.plot(kpts[:,1], cst.Ry * (epsk[:,i]))

   ax.set_ylim(cst.Ry *Emin, cst.Ry *Emax)
   plt.show()
   exit()
   '''

   toc = time.perf_counter()
   tools.PrintTime(tic,toc,"band structure")

   band_range = [19, 20]

   tic = time.perf_counter()
   # pol = GetPol_p(alpha)
   # pol = geo.GetPol_p_i()
   pol = geo.GetPol_s_i()
   inten_p = np.zeros([nk, len(ebind)])
   for i, en in enumerate(ebind):
      inten_p[:,i] = GetIntensity(epsk,vectk,en,mel_at[:,:,:,i],pol,eta,mu,Eshift=0.,band_range=band_range)

   PlotIntensity(kpts[:,0],ebind,inten_p,fac=1.0,units='au')
#----------------------------------------------------------------------
if __name__ == '__main__':
    main(sys.argv[1:])
