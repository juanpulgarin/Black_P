import numpy as np
import h5py
import constants as cst
#------------------------------------------------------------
def GetHR(fname):
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

	data = np.loadtxt(fname,skiprows=13)

	IRv = np.arange(0,nR)
	Iwv = np.arange(0,nwan*nwan)

	irvec = np.zeros([2*nR-1,3],dtype=np.int_)
	irvec[0:nR,0:2] = data[nwan*nwan*IRv,0:2].astype(int)
	irvec[nR:,0:2] = -irvec[1:nR,0:2]

	ham_r = np.zeros([nwan,nwan,2*nR-1],dtype=np.complex_)

	for iR in range(nR):
		x = data[nwan*nwan*iR:nwan*nwan*iR+nwan*nwan,5:7]
		ham_r[:,:,iR] = np.reshape(x[:,0] + 1j*x[:,1], [nwan,nwan])

	for iR in range(nR-1):
		ham_r[:,:,nR+iR] = np.conj(ham_r[:,:,iR].T)

	return irvec, ham_r
#------------------------------------------------------------
def GetHk(fname):
	with open(fname, 'r') as f:
		lines = f.readlines()
		for i,line in enumerate(lines):
			if line.find("#nk      nwan") != -1:
				ili = i + 1

	s = lines[ili].split()
	nk = int(s[0])
	nwan = int(s[1])		

	print("nk = ", nk)
	print("nwan = ", nwan)

	data = np.loadtxt(fname,skiprows=14)	

	Ikv = np.arange(0,nk)

	kpts = data[nwan*nwan*Ikv,0:2] - 0.5

	ham_k = np.zeros([nwan,nwan,nk],dtype=np.complex_)

	for ik in range(nk):
		x = data[nwan*nwan*ik:nwan*nwan*ik+nwan*nwan,5:7]
		ham_k[:,:,ik] = np.reshape(x[:,0] + 1j*x[:,1], [nwan,nwan])

	return kpts, ham_k / cst.Ry
#------------------------------------------------------------
def Transform_R(kpts,ham_k,nk1,nk2,nk3):
	nwan = ham_k.shape[0]
	nR = ham_k.shape[-1]

	ham_r = np.zeros([nwan,nwan,nR], dtype=np.complex_)

	for i in range(nwan):
		for j in range(nwan):
			x = np.reshape(ham_k[i,j,:], [nk1,nk2,nk3])
			y = np.fft.fftn(x)
			ham_r[i,j,:] = np.reshape(y, [nR]) / nR

	xr = np.fft.fftfreq(nk1,d=1. / nk1).astype(int)
	yr = np.fft.fftfreq(nk2,d=1. / nk2).astype(int)
	zr = np.fft.fftfreq(nk3,d=1. / nk3).astype(int)
	X1, X2, X3 = np.meshgrid(xr, yr, zr, indexing='ij')
	# X1, X2 = np.meshgrid(xr, yr)
	irvec = np.zeros([nR, 3], dtype=np.int_)
	irvec[:,0] = np.reshape(X1, [nR])
	irvec[:,1] = np.reshape(X2, [nR])
	irvec[:,2] = np.reshape(X2, [nR])

	return irvec, ham_r
#------------------------------------------------------------
def SaveHamTB(fname,irvec,ham_r,lat,pos):

	nwan = ham_r.shape[0]
	nR = ham_r.shape[-1]

	f = h5py.File(fname, "w")

	f.attrs['atomic_units'] = 1
	f.attrs['num_wann'] = nwan
	f.attrs['nrpts'] = nR
	f.attrs['pos_stored'] = 1
	f.attrs['coords_stored'] = 1

	ndegen = np.ones(nR)
	f.create_dataset('ndegen', data=ndegen)
	f.create_dataset('real_lattice', data=lat.T)
	f.create_dataset('irvec', data=irvec.T)
	f.create_dataset('ham_r_real', data=np.real(ham_r).T)
	f.create_dataset('ham_r_imag', data=np.imag(ham_r).T)

	# pos_r = np.zeros([nwan,nwan,nR,3],dtype=np.complex_)
	pos_r = np.zeros([nwan,nwan,3,nR],dtype=np.complex_)
	for idir in range(3):
		pos_r[:,:,0,idir] = np.diag(pos[:,idir])

	f.create_dataset('pos_r_real', data=np.real(pos_r).T)
	f.create_dataset('pos_r_imag', data=np.imag(pos_r).T)

	f.create_dataset('coords', data=pos.T)

	f.close()
#------------------------------------------------------------
def Import_BlackP():


	file_ham = "./data/BlackP/Hk.dat.qpgw"
	kpts, ham_k = GetHk(file_ham)

	nk1, nk2, nk3 = 12, 12, 8
	irvec, ham_r = Transform_R(kpts,ham_k,nk1,nk2,nk3)

	a = 3.31067978852923 / cst.aB
	b = 4.58545824667620 / cst.aB
	c = 10.9888937548268 / cst.aB


	lat = np.zeros([3,3])
	lat[0,:] =  np.array([a,  0.0, 0.0])
	lat[1,:] =  np.array([0.0,  b, 0.0])
	lat[2,:] =  np.array([0.0, 0.0,  c])

	nwan = ham_r.shape[0]

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

	pos = np.zeros([nwan,3])
	for iat,frac in enumerate(coords):
		pos[iat*4:iat*4 + 4,:] = np.sum(frac[None,:] * lat, axis=0)
		print(iat, frac)
	print(pos)
	fout = "data/BlackP/Hqpgw.h5"

	SaveHamTB(fout,irvec,ham_r,lat,pos)

#------------------------------------------------------------
def main():
	Import_BlackP()
#------------------------------------------------------------
if __name__ == '__main__':
	main()
