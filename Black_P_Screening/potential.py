import numpy as np
import scipy.fftpack as sfft
import matplotlib.pyplot as plt
import h5py
import pickle
import constants as cst
from LatticeInteraction import LattInter
#------------------------------------------------------------
def GetCRPA(fname,filter_thresh=0.0):
	with open(fname, 'r') as f:
		lines = f.readlines()
		for i,line in enumerate(lines):
			if line.find("#nR      nwan") != -1:
				ili = i + 1

	s = lines[ili].split()
	nR = int(s[0])
	nwan = int(s[1])

	print("nR = ", nR)
	print("nwan = ", nwan)

	data = np.loadtxt(fname,skiprows=14)

	IRv = np.arange(0,nR)

	irvec = data[nwan*nwan*IRv,0:3].astype(int)

	Vint_r = np.zeros([nwan,nwan,nR])

	for iR in range(nR):
		x = data[nwan*nwan*iR:nwan*nwan*iR+nwan*nwan,7]
		Vint_r[:,:,iR] = np.reshape(x, [nwan,nwan]) / cst.Ry
		# Vint_r[:,:,iR] = np.reshape(x, [nwan,nwan])

	if filter_thresh > 0.0:
		for i in range(nwan):
			for j in range(nwan):
				x = Vint_r[i,j,:]
				Vint_r[i,j,:] = Filter(filter_thresh, x)

	return irvec, Vint_r
#------------------------------------------------------------------------
def Transform_q(nk1,nk2,Vint_r):

	nwan = Vint_r.shape[0]
	nR = Vint_r.shape[-1]

	Vint_q = np.zeros([nwan,nwan,nR], dtype=np.complex_)

	for i in range(nwan):
		for j in range(nwan):
			x = np.reshape(Vint_r[i,j,:], [nk1,nk2])
			# y = np.fft.ifft2(x)
			y = sfft.ifft2(x)
			y_shift = sfft.fftshift(y)
			Vint_q[i,j,:] = np.reshape(y_shift, [nR])

	return Vint_q
#------------------------------------------------------------------------
def Transform_r(nk1,nk2,Vint_q):
	nwan = Vint_q.shape[0]
	nR = Vint_q.shape[-1]

	Vint_r = np.zeros([nwan,nwan,nR])

	for i in range(nwan):
		for j in range(nwan):
			x = np.reshape(Vint_q[i,j,:], [nk1,nk2])
			x_shift = sfft.ifftshift(x)
			# y = np.fft.ifft2(x)
			y = sfft.fft2(x_shift)
			Vint_r[i,j,:] = np.reshape(np.real(y), [nR])

	return Vint_r
#------------------------------------------------------------------------
def ConstructInteraction(irvec,Vint_r):
	norb = Vint_r.shape[0]
	nR = Vint_r.shape[-1]

	coords = np.array([
		[0.75000,   0.41344,   0.90284],
		[0.25000,   0.08656,   0.90284],
		[0.25000,   0.58656,   0.59716],
		[0.75000,   0.91344,   0.59716],
		[0.25000,   0.41344,   0.40284],
		[0.75000,   0.08656,   0.40284],
		[0.75000,   0.58656,   0.09716],
		[0.25000,   0.91344,   0.09716]
		])

	nat = coords.shape[0]

	orb = np.zeros([norb,3])
	for j in range(nat):
		orb[j*4:j*4+4,:] = coords[j]

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
#------------------------------------------------------------------------
def Filter(rc,x):
	a = 2 / rc
	b = (1 - a*rc) / rc**2

	y = np.zeros_like(x)
	Ix1, = np.where((x > 0.) & (x < rc))
	y[Ix1] = a*x[Ix1]**2 + b*x[Ix1]**3
	Ix2, = np.where(x >= rc)
	y[Ix2] = x[Ix2]

	return y
#------------------------------------------------------------------------
def Import_cRPA():

    cdm3 = 8.
    eps_sub = 1.

    file_crpa = "data/BlackP/cRPA_Uiijj.dat"
    irvec, Vint_r = GetCRPA(file_crpa, filter_thresh=0.001)

    Uhubb_r = np.array([ Vint_r[i,i,0] for i in range(Vint_r.shape[0]) ])
    print("U (real space) = ", Uhubb_r)

    Vint = ConstructInteraction(irvec,Vint_r)
    fname = "data/BlackP/Vbare_crpa_michael.h5"
    Vint.SaveToHDF5(fname)
#------------------------------------------------------------------------
def main():
    Import_cRPA()
#------------------------------------------------------------
if __name__ == '__main__':
	main()
