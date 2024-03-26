import numpy as np
import scipy.linalg as la

#----------------------------------------------------------------------
def GenKgrid3D_cart(kxr,kyr,kzr,nk1,nk2,nk3):
   kxs = np.linspace(kxr[0], kxr[1], nk1)
   kys = np.linspace(kyr[0], kyr[1], nk2)
   kzs = np.linspace(kzr[0], kzr[1], nk3)
   X, Y, Z = np.meshgrid(kxs, kys, kzs)
   kpts = np.zeros([nk1*nk2*nk3,3])
   kpts[:,0] = np.reshape(X,[nk1*nk2*nk3])
   kpts[:,1] = np.reshape(Y,[nk1*nk2*nk3])
   kpts[:,2] = np.reshape(Y,[nk1*nk2*nk3])
   return kpts
#----------------------------------------------------------------------
def FracToCart3D(kcart,b1,b2,b3):
   M = la.inv(np.array([b1, b2, b3]).T)
   kred = np.zeros([kcart.shape[0],3])
   kred[:,0:3] = np.einsum('rx,ix->ir', M, kcart[:,0:3])
   return kred
