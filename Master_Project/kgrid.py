import numpy as np
import scipy.linalg as la

import honeycomb as lat

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
#----------------------------------------------------------------------
def K_PATH(k_points_path,n_kpts):
    new_path = np.zeros((3,(np.shape(k_points_path)[0]-1)*n_kpts))

    for i in range(np.shape(k_points_path)[0]-1):
        eje_x = np.linspace(k_points_path[i,0],k_points_path[i+1,0],n_kpts+1)[0:-1]
        eje_y = np.linspace(k_points_path[i,1],k_points_path[i+1,1],n_kpts+1)[0:-1]
        eje_z = np.linspace(k_points_path[i,2],k_points_path[i+1,2],n_kpts+1)[0:-1]

        new_path[0,i*n_kpts:(i+1)*n_kpts] = eje_x
        new_path[1,i*n_kpts:(i+1)*n_kpts] = eje_y
        new_path[2,i*n_kpts:(i+1)*n_kpts] = eje_z

    return new_path
#----------------------------------------------------------------------
 def GenKgrid(nk1,nk2,nk3,path):
    name_lat = path+'Lattice'
    orbitales,parameters = Reading.ReadLattice(name_lat+'.h5')

    a = parameters[0] / cst.aB
    b = parameters[1] / cst.aB
    c = parameters[2] / cst.aB
    a1, a2, a3 = lat.GetRealLatt(a,b,c)
    b1, b2, b3 = lat.GetRecLatt(a1,a2,a3)
    kp = [0.0,0.0,0.0]
    delta_x = .2
    delta_y = .2
    delta_z = .2
    kxr = [kp[0]-delta_x,kp[0]+delta_x]
    kyr = [kp[1]-delta_y,kp[1]+delta_y]
    kzr = [kp[2]-delta_z,kp[2]+delta_z]
    kcart = kgrid.GenKgrid3D_cart(kxr,kyr,kzr,nk1,nk2,nk3)
    kpts = kgrid.FracToCart3D(kcart,b1,b2,b3)

    return kpts

