import numpy as np
import h5py
#------------------------------------------------------------------------
class LattInter():
	#========================================
	def __init__(self,orb):

		self.ndim = orb.shape[1]
		self.norb = orb.shape[0]
		self.orb_pos = orb
		self.nhop = 0

		# self.hop_orb = np.array([[]],dtype=np.int_)
		self.hop_vec = np.array([[]],dtype=np.int_)
		self.Vlatt = np.array([],dtype=np.float_)
		self.U =  np.zeros(self.norb)

	#========================================
	def set_hubbard(self,U):
		self.U = U
	#========================================
	def set_V(self,V,latt_vec,transpose=True):
		if self.nhop == 0:
			self.hop_vec = np.array([latt_vec])
			if transpose:
				self.Vlatt = np.array([V.T])
			else:
				self.Vlatt = np.array([V])
		else:
			self.hop_vec = np.append(self.hop_vec, [latt_vec], axis=0)	
			if transpose:
				self.Vlatt = np.append(self.Vlatt, [V.T], axis=0)
			else:
				self.Vlatt = np.append(self.Vlatt, [V], axis=0)

		self.nhop += 1
	#========================================
	def GetVq(self,qred):
		Vq = np.zeros([self.norb,self.norb],dtype=np.complex_)
		for i in range(self.nhop):
			ind_R = self.hop_vec[i]
			phase = np.exp(2.0j * np.pi * np.dot(qred,ind_R))

			Vq[:,:] += phase * self.Vlatt[i,:,:]
			# if add_herm:
			# 	Vq[orb2, orb1] += np.conj(phase * self.Vlatt[i])

		return Vq
	#========================================	
	def SaveToHDF5(self,fname):
		f = h5py.File(fname, 'w')

		f.attrs['ndim'] = self.ndim
		f.attrs['norb'] = self.norb
		f.attrs['nhop'] = self.nhop
		f.create_dataset('orb_pos', data=self.orb_pos)
		# f.create_dataset('hop_orb', data=self.hop_orb)
		f.create_dataset('hop_vec', data=self.hop_vec)
		f.create_dataset('uhubb', data=self.U)
		f.create_dataset('vlatt', data=self.Vlatt)

		f.close()
	#========================================	
	def ReadFromHDF5(self,fname):
		f = h5py.File(fname, 'r')

		self.ndim = f.attrs['ndim']
		self.norb = f.attrs['norb']
		self.nhop = f.attrs['nhop']

		# self.hop_orb = np.array(f['hop_orb'])
		self.hop_vec = np.array(f['hop_vec'])
		self.U = np.array(f['uhubb'])
		self.Vlatt = np.array(f['vlatt'])

		f.close()
#------------------------------------------------------------------------