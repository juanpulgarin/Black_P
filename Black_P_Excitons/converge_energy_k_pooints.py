import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt


k1 = np.array([12,16,24,32])
k3=np.array([8,12,16,24,32])


energies = []
k_points = []
for kk1 in range(len(k1)):
	for kk3 in range(kk1+2):
		#print(str(k1[kk1])+"x"+str(k1[kk1])+"x"+str(k3[kk3]))
		name = str(k1[kk1])+'x'+str(k1[kk1])+'x'+str(k3[kk3])+'.log'
		#print(name)
		result = subprocess.run(['bash','./log/read.sh', './log/'+name, 'E(1)'],capture_output=True,text=True)
		energies.append(float(result.stdout.strip()))
		k_points.append(k1[kk1]**2*k3[kk3])


plt.figure()
plt.plot(k_points,energies,".")
plt.show()


