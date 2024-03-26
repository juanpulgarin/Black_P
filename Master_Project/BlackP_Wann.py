#!/opt/psi/Programming/Python/3.9.10/bin/python3
#===================================================#
#SBATCH --cluster=merlin6
#SBATCH --partition=hourly
#SBATCH -V
#SBATCH --job-name=gra
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/gra_%A_%a.out
#===================================================#

import sys
import os
import numpy as np
sys.path.append('./')
import constants as cst
import Reading_Scripts as Reading
import kgrid #------------------------------
import honeycomb as lat #------------------------------

home = os.environ['HOME']
dynw90_root = home + '/dynamics-w90/'
sys.path.append(dynw90_root + 'python_utils/')
from wann_calc import WannierCalculation
#----------------------------------------------------------------------
def GenKgrid(nk1,nk2,nk3):
    system="BlackP/"
    path="./"
    name_lat = path+'data/'+system+'Lattice'
    orbitales,parameters = Reading.ReadLattice(name_lat+'.h5')
    
    a = parameters[0] / cst.aB
    b = parameters[1] / cst.aB
    c = parameters[2] / cst.aB
    a1, a2, a3 = lat.GetRealLatt(a,b,c)
    b1, b2, b3 = lat.GetRecLatt(a1,a2,a3)
    kp = [0.0,0.0,0.0]
    delta_x = .06
    delta_y = .06
    delta_z = .06
    kxr = [kp[0]-delta_x,kp[0]+delta_x]
    kyr = [kp[1]-delta_y,kp[1]+delta_y]
    kzr = [kp[2]-delta_z,kp[2]+delta_z]
    kcart = kgrid.GenKgrid3D_cart(kxr,kyr,kzr,nk1,nk2,nk3)
    kpts = kgrid.FracToCart3D(kcart,b1,b2,b3)
    file_kpts = "data/kgrid/cart_kxr{}_{}_kyr{}_{}_kzr{}_{}_nk{}x{}x{}.dat".\
           format(kxr[0],kxr[-1],kyr[0],kyr[-1],kzr[0],kzr[-1],nk1,nk2,nk3)
    if not os.path.isfile(file_kpts):
        np.savetxt(file_kpts, np.c_[kpts])
    return file_kpts


def run_calc(mpicmd,nk1,nk2,nk3,gauge=0,debug_mode=True):

    # parameters
    mu = 0.0 / cst.Ry
    Temp = 20 # K
    #beta = 1.0 / (Temp * cst.kB)
    beta = 1.0 * 10**5
    file_ham = "./data/BlackP/Hqpgw_new.h5"
    # k-grid
    file_kpts = GenKgrid(nk1,nk2,nk3)
    
    calc = WannierCalculation(dynw90_root,mpicmd=mpicmd)
    
    calc.SetHamiltonian(mu,beta,file_ham,gauge,FixMuChem=True)
    calc.SetKPTS('list', file_kpts=file_kpts)
    
    calc.SetOutput(calc_orbweight=True,calc_berry=True,calc_metric=True,calc_oam=False,calc_evecs=True,write_velocity=True,calc_spin=False,calc_spin_berry=False,berry_valence=False,gauge=0)
    prefix = "BlackP_grid_list_nk{}x{}x{}".format(nk1,nk2,nk3)
    calc.Run(prefix,debug_mode=debug_mode)



#----------------------------------------------------------------------
def main(mpicmd):

    nk1 = 16
    nk2 = 16
    nk3 = 16
    gauge=0
        
    run_calc(mpicmd,nk1,nk2,nk3,gauge,debug_mode=True)
#----------------------------------------------------------------------
if __name__ == '__main__':
    mpicmd = 'srun'
    os.environ['OMP_NUM_THREADS']='1'
    main(mpicmd)