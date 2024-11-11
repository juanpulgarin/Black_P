import numpy as np
import scipy.linalg as la
import pickle
import time
#-------------------------------------------------------------------
def PrintTag(s,l=50):
   len_s = len(s) + 6
   len_left = (l - len_s)//2 - 1
   print(len_left*" ","++ ",s," ++")
#-------------------------------------------------------------------
def PrintTime(tic,toc,tag):
   print("Time[",tag,f"] = {toc-tic:0.4} s")
#----------------------------------------------------------------------
def MirrorY(three_d=False):
	if three_d:
		R = np.diag([1.0, -1.0, 1.0])
	else:
		R = np.diag([1.0, -1.0])
	return R
#----------------------------------------------------------------------
def MirrorX(three_d=False):
	if three_d:
		R = np.diag([-1.0, 1.0, 1.0])
	else:
		R = np.diag([-1.0, 1.0])
	return R
#----------------------------------------------------------------------
def Rotz_2d(theta):
	R = np.zeros([2,2])
	R[0,0] = np.cos(theta)
	R[0,1] = -np.sin(theta)
	R[1,0] = -R[0,1]
	R[1,1] = R[0,0]
	return R
#----------------------------------------------------------------------
def Rotz(theta):
	R = np.zeros([3,3])
	R[0,0] = np.cos(theta)
	R[0,1] = -np.sin(theta)
	R[1,0] = -R[0,1]
	R[1,1] = R[0,0]
	R[2,2] = 1.0
	return R
#----------------------------------------------------------------------
def Rotx(phi):
    R = np.zeros([3,3])

    R[0,0] = 1.0
    R[1,1] = np.cos(phi)
    R[1,2] = -np.sin(phi)
    R[2,1] = np.sin(phi)
    R[2,2] = np.cos(phi)

    return R
#----------------------------------------------------------------------
def ReadOrbdata(fname):
	f = open(fname, "rb")
	orb_data = pickle.load(f)
	f.close()

	return orb_data
#----------------------------------------------------------------------
