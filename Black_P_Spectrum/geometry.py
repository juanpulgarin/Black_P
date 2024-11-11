import numpy as np
#----------------------------------------------------------------------
class LightGeometry(object):
	"""docstring for LightGeometry"""
	def __init__(self, alpha, theta, epsr, epsi=0.0):
		super(LightGeometry, self).__init__()
		self.alpha = alpha
		self.theta = theta
		self.epsr = epsr
		self.ca = np.cos(self.alpha)
		self.sa = np.sin(self.alpha)

		nr = np.sqrt(epsr - 1j * epsi)
		# print(nr)	

		self.sin_th = np.sin(alpha) / nr
		self.cos_th = np.sqrt(1. - self.sin_th**2)
		#======================================================================================
		self.rot_plane = np.array(	[[ np.cos(self.theta),  np.sin(self.theta),     0.0],
								[- np.sin(self.theta),  np.cos(self.theta),     0.0],
								[       0.0,                           0.0,     1.0]])
		#======================================================================================
		# self.ts = 2*self.ca / (self.ca + self.nr * self.cos_th)
		# self.tp = 2*self.ca / (self.nr * self.ca + self.cos_th)
		# self.rs = self.ts - 1.0
		# self.rp = self.nr * self.tp - 1.0
		# self.rs = (self.ca - self.nr * self.cos_th) / (self.ca + self.nr * self.cos_th)
		# self.rp = (self.nr * self.ca - self.cos_th) / (self.nr * self.ca + self.cos_th)
		m2s = np.sqrt(nr**2 - self.sa**2)
		self.ts = 2*self.ca / (self.ca + m2s)
		self.tp = 2*nr*self.ca / (nr**2 + m2s)
		self.rs = (self.ca - np.sqrt(nr**2 - self.sa**2)) / (self.ca + np.sqrt(nr**2 - self.sa**2))
		self.rp = (nr**2*self.ca - np.sqrt(nr**2 - self.sa**2)) / (nr**2*self.ca + np.sqrt(nr**2 - self.sa**2))
	#===========================================================
	def GetPoynting_i(self):
		s = self.rot_plane @ np.array([self.sa, 0.0, -self.ca])
		return s
	#===========================================================
	def GetPoynting_r(self):
		s = self.rot_plane @ np.array([self.sa, 0.0, self.ca])
		return s
	#===========================================================
	def GetPoynting_t(self):
		s = self.rot_plane @ np.array([self.sin_th, 0.0, -self.cos_th])
		return s
	#===========================================================
	def GetPol_p_i(self):
		p = self.rot_plane @ np.array([self.ca, 0.0, self.sa])
		return p
	#===========================================================
	def GetPol_p_r(self):
		p = self.rot_plane @ self.rp * np.array([-self.ca, 0.0, self.sa])
		return p
	#===========================================================
	def GetPol_p_t(self):
		p = self.rot_plane @ self.tp * np.array([self.cos_th, 0.0, self.sin_th])
		return p		
	#===========================================================
	def GetPol_s_i(self):
		return self.rot_plane @ np.array([0.0, 1.0, 0.0])
	#===========================================================
	def GetPol_s_r(self):
		return self.rot_plane @ self.rs * np.array([0.0, 1.0, 0.0])
	#===========================================================
	def GetPol_s_t(self):
		return self.rot_plane @ self.ts * np.array([0.0, 1.0, 0.0])

#----------------------------------------------------------------------
