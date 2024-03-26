import numpy as np
from LatticeInteraction import LattInter
import h5py
import pickle

def ReadBands(fname):
    f = h5py.File(fname, "r")
    
    epsk = np.array(f['epsk'])
    
    f.close()
    return epsk

def ReadVint_realspace(fname):
    f = h5py.File(fname, "r")
    
    hop_vec = np.array(f['hop_vec'])
    uhubb = np.array(f['uhubb'])
    vlatt = np.array(f['vlatt'])
    
    f.close()
    return hop_vec, uhubb, vlatt

def ReadVint_reciprocalspace(fname):
    f = h5py.File(fname, "r")
    
    vlatt_imag = np.array(f['imag'])
    kdims      = np.array(f['kdims'])
    vlatt_real = np.array(f['real'])
    uhubb      = np.array(f['uhubb'])
    
    f.close()
    return kdims, uhubb, vlatt_real+1.0j*vlatt_imag

def ReadLattice(fname):
    f = h5py.File(fname, "r")
    
    orb = np.array(f['orb'])
    lat = np.array(f['lat'])
    
    f.close()
    return orb,lat

def Export_lattspace(irvec,lat,V_r,fname):
    data = {'irvec': irvec, 'lat': lat, 'Vr': V_r}
    f = open(fname, "wb")
    pickle.dump(data, f)
    f.close()
    
def ReadKpts(fname):
    f = h5py.File(fname, "r")
    kpts = np.array(f['kpts'])
    f.close()
    return kpts

def Joining(irvec,Vint_r,Uint_r,orbitales,lat):
    xs = irvec[:,0]
    ys = irvec[:,1]
    zs = irvec[:,2]

    a1, a2, a3 = lat[0,:], lat[1,:], lat[2,:]

    rvecs = xs[:,None] * a1[None,:] + ys[:,None] * a2[None,:]+ zs[:,None] * a3[None,:]
    ds = np.sqrt(rvecs[:,0]**2 + rvecs[:,1]**2 + rvecs[:,2]**2)

    Ix = np.argsort(ds)

    for k in range(np.shape(Vint_r)[1]):

        Vint_r[np.where(abs(ds)==0)[0][0],k,k]=Uint_r[k]

    return Vint_r.T

def ConstructInteraction(orb,irvec,Vint_r):
    norb = Vint_r.shape[0]
    nR = Vint_r.shape[-1]


    vint = LattInter(orb)
    Uvec = np.diag(Vint_r[:,:,0])
    vint.set_hubbard(Uvec)

    Vint_r0 = np.array(Vint_r[:,:,0])
    for i in range(norb):
        Vint_r0[i,i] = 0.0

    vint.set_V(Vint_r0, [0, 0, 0], transpose=True)

    for ir in range(1,nR):
        n1, n2, n3 = irvec[ir,0], irvec[ir,1], irvec[ir,2]
        vint.set_V(Vint_r[:,:,ir], [n1, n2, n3], transpose=True)



    return vint
