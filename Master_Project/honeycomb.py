import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
def GetRealLatt(a,b,c):
	a1 = np.array([a,  0.0, 0.0])
	a2 = np.array([0.0,  b, 0.0])
	a3 = np.array([0.0, 0.0,  c])
	return a1,a2,a3
#----------------------------------------------------------------------
def GetRecLatt(a1,a2,a3):
	V = np.dot(a1,np.cross(a2, a3) )
	b1 = 2*np.pi/V * np.cross(a2, a3)
	b2 = 2*np.pi/V * np.cross(a3, a1)
	b3 = 2*np.pi/V * np.cross(a1, a2)
	return b1,b2,b3
#----------------------------------------------------------------------
def FracToCart(kpt,b1,b2,b3):
	return kpt[0] * b1 + kpt[1] * b2 +  kpt[2] * b3
#----------------------------------------------------------------------
def CartToFrac(kpt,b1,b2,b3):
	M = np.array([b1, b2, b3]).T
	return la.solve(M, kpt)
#----------------------------------------------------------------------
"""def GetDiracPoints(alat, rot_ang=0.0):
	a1, a2 = GetRealLatt(alat, rot_ang=rot_ang)
	brec1, brec2 = GetRecLatt(a1,a2)

	Ks = np.zeros([6,2])

	Ks[0,:] = -1.0/3.0 * brec1 + 2.0/3.0 * brec2
	Ks[1,:] = 1.0/3.0 * brec1 + -2.0/3.0 * brec2
	Ks[2,:] = 2.0/3.0 * brec1 + -1.0/3.0 * brec2
	Ks[3,:] = -2.0/3.0 * brec1 + 1.0/3.0 * brec2
	Ks[4,:] = 1.0/3.0 * brec1 + 1.0/3.0 * brec2
	Ks[5,:] = -1.0/3.0 * brec1 - 1.0/3.0 * brec2

	return Ks"""
#----------------------------------------------------------------------
"""def DrawBZ(alat,ax,rot_ang=0,show_points=True,show_labels=False,ls=':'):

	Ks = GetDiracPoints(alat, rot_ang=rot_ang)

	K1 = Ks[0,:]
	K2 = Ks[1,:]
	K3 = Ks[2,:]
	K4 = Ks[3,:]
	K5 = Ks[4,:]
	K6 = Ks[5,:]

	if show_points:
		ax.scatter([K1[0],K2[0],K3[0],K4[0],K5[0],K6[0]],[K1[1],K2[1],K3[1],K4[1],K5[1],K6[1]],c='k',s=20)

	ax.plot([K6[0],K4[0]],[K6[1],K4[1]],c='k',ls=ls)
	ax.plot([K2[0],K6[0]],[K2[1],K6[1]],c='k',ls=ls)
	ax.plot([K1[0],K5[0]],[K1[1],K5[1]],c='k',ls=ls)
	ax.plot([K4[0],K1[0]],[K4[1],K1[1]],c='k',ls=ls)
	ax.plot([K5[0],K3[0]],[K5[1],K3[1]],c='k',ls=ls)
	ax.plot([K2[0],K3[0]],[K2[1],K3[1]],c='k',ls=ls)

	if show_labels:
		ax.text(K1[0], K1[1], r"K$_1 $", color='red', fontsize=14)
		ax.text(K2[0], K2[1], r"K$_2 $", color='red', fontsize=14)
		ax.text(K3[0], K3[1], r"K$_3 $", color='red', fontsize=14)
		ax.text(K4[0], K4[1], r"K$_4 $", color='red', fontsize=14)
		ax.text(K5[0], K5[1], r"K$_5 $", color='red', fontsize=14)
		ax.text(K6[0], K6[1], r"K$_6 $", color='red', fontsize=14)"""
#----------------------------------------------------------------------
"""if __name__ == '__main__':
	alat = 6.27389

	fig, ax = plt.subplots()
	DrawBZ(alat,ax,ls='-')
	ax.set_xlim(-1.5,1.5)
	ax.set_ylim(-1.5,1.5)
	ax.set_aspect(1.0)
	plt.show()"""

