import os
import sys
import numpy as np
import h5py
import time
# ---- local modules -----
import tools
import constants as cst
from geometry import LightGeometry
from orbdata import GetOrbData_tb
from pes_intensity import CalcAtomicMatrixElements
# ---- external modules -----
from python_w90 import wann90_tb
#-------------------------------------------------------------
def GenMatrixElements(ws,kpath,polarization,bands,photon,project_band=-1):
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
        kpts[:,0] = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])
    else:
        kpts[:,1] = np.linspace(kpath['kmin'], kpath['kmax'], kpath['nk'])

    orb = GetOrbData_tb(rint[0], rint[1], rint[2])
        
    ws_shift = ws - Phiw
    mel_at = CalcAtomicMatrixElements(orb,ws_shift,wphot,Vin,eta,kpts)

    ham = wann90_tb("data/blackp.h5", from_hdf5=True)

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
        epsk[i, :] = eps[bands]
        rotk[i, :, :] = evect[:, bands]

    # rotate matrix elements
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
#-------------------------------------------------------------
def main():

    alpha = 54

    scissor_index = 20
    scissor_energy = 0.27 / cst.Ry
    Eshift = -6.595 / cst.Ry

    ibnd_min = 19
    ibnd_max = 20
    bands = np.arange(ibnd_min, ibnd_max+1, step=1)

    nkz = 41

    tic = time.perf_counter()

    wmin, wmax = -1.2 / cst.Ry, 1.2 / cst.Ry
    Nepe = 401
    ws = np.linspace(wmin, wmax, Nepe)

    kpath = {
        'nk': 128,
        'kmin': -0.15,
        'kmax': 0.15
    }
    
    photon = {
        'wphot': 6.2,
        'alpha': alpha,
        'Vin': 0.0,
        'Phiw': 4.95409,
        'Eshift': 0.,
        'rint': [0.2, 1.0, 2.0],
        'eta': 0.002
    }

    # probe
    geo = LightGeometry(alpha/180. * np.pi, 1.0)
    pol_p = geo.GetPol_p_i()
    pol_s = geo.GetPol_s_i()

    kpath['dir'] = 'y'
    polarization = {'tag': 's', 'vector': pol_s}
    GenMatrixElements(ws,kpath,polarization,bands,photon)

    polarization = {'tag': 'p', 'vector': pol_p}
    GenMatrixElements(ws,kpath,polarization,bands,photon)
 
    kpath['dir'] = 'x'
    polarization = {'tag': 's', 'vector': pol_s}
    GenMatrixElements(ws,kpath,polarization,bands,photon)

    polarization = {'tag': 'p', 'vector': pol_p}
    GenMatrixElements(ws,kpath,polarization,bands,photon)
#-------------------------------------------------------------            
if __name__ == "__main__":
    main()
    
