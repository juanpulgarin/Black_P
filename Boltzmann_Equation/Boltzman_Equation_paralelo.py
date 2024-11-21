#from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from scipy.integrate import solve_ivp,odeint,trapezoid
import time
from typing import Tuple
import h5py

from joblib import Parallel, delayed

import constants as cst

#============================================================================
#####-------------------------Optimization-----------------------------------
def Green_optim_rid(G, k):
    rng = np.random.default_rng()
    oversampling = int(k)
    p = k + oversampling
    print("number of samples = ", p)
    idx = rng.choice(G.shape[1],replace=False ,size=p)
    AS = G[:,idx]
    _, R, P = sla.qr(AS, pivoting=True,
        mode='economic',
        check_finite=False)
    R_k = R[:k,:k]
    _cols = P[:k]
    cols = idx[_cols]
    C = AS[:,_cols]
    # Rk_Rk = R_k.T @ R_k
    # b = C.T @ G
    Rk_Rk = np.conj(R_k.T) @ R_k
    b = np.conj(C.T) @ G
    Z = sla.solve(Rk_Rk, b, overwrite_b=True)
    approx = C @ Z
    return approx , cols , Z

#approz,cols, Z = Green_optim_rid(matrix nxn,reduccion, maximo valor es n/2)

#============================================================================
#####--------------------------Grids of Q------------------------------------
def create_Q(n :int, Δ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dimensiones = len(Δ)

    if dimensiones == 1:
        Q,Q_reduced = create_Q_1D(n,Δ)

    if dimensiones == 2:
        Q,Q_reduced = create_Q_2D(n,Δ)

    if dimensiones == 3:
        Q,Q_reduced = create_Q_3D(n,Δ)


    Q0_reduced = np.zeros(np.shape(Q_reduced)[0])
    Q0         = np.zeros(np.shape(Q)[0])
    Q0_reduced[0]           = 1.0
    Q0[n**dimensiones // 2] = 1.0

    return Q,Q_reduced,Q0,Q0_reduced

def create_Q_3D(n :int, Δ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-Δ[0],Δ[0],n)
    y = np.linspace(-Δ[1],Δ[1],n)
    z = np.linspace(-Δ[2],Δ[2],n)
    xv,yv,zv = np.meshgrid(x,y,z)

    Q = np.zeros((n**3,3))
    for i in range(n**3):
        Q[i,0] = xv.reshape(n**3)[i]
        Q[i,1] = yv.reshape(n**3)[i]
        Q[i,2] = zv.reshape(n**3)[i]

    Ik, = np.where( (Q[:,0] >= 0) & (Q[:,1] >= 0) & (Q[:,2] >= 0) )
    Q_reduced = Q[Ik,:]

    return Q,Q_reduced

def create_Q_2D(n :int, Δ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-Δ[0],Δ[0],n)
    y = np.linspace(-Δ[1],Δ[1],n)
    xv,yv = np.meshgrid(x,y)

    Q = np.zeros((n**2,2))
    for i in range(n**2):
        Q[i,0] = xv.reshape(n**2)[i]
        Q[i,1] = yv.reshape(n**2)[i]

    Ik, = np.where( (Q[:,0] >= 0) & (Q[:,1] >= 0) )
    Q_reduced = Q[Ik,:]

    return Q,Q_reduced

def create_Q_1D(n :int, Δ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-Δ[0],Δ[0],n)

    Q = np.zeros((n**1,1))
    for i in range(n**1):
        Q[i,0] = x[i]

    Ik, = np.where( (Q[:,0] >= 0) )
    Q_reduced = Q[Ik,:]

    return Q,Q_reduced

"""@njit
def flipping_numba(vector):
    num_nonzero = np.sum(vector != 0)
    num_combinations = 2 ** num_nonzero  # Number of sign combinations for non-zero elements
    combinations = np.zeros((num_combinations, len(vector)))

    for i in range(num_combinations):
        comb = np.copy(vector)
        flip_index = 0
        for j in range(len(vector)):
            if vector[j] != 0:
                if (i >> flip_index) & 1:  # Flip the sign if the corresponding bit is set
                    comb[j] = -comb[j]
                flip_index += 1
        combinations[i, :] = comb

    return combinations"""

def flipping(vector):
    sign_options = [[x, -x] if x != 0 else [x] for x in vector]

    combinations = list(itertools.product(*sign_options))
    return  [np.array(comb) for comb in combinations]

#============================================================================
#####--------------------Density of States (old)-----------------------------
def DOS(sigma :float,omega :float,Exciton_Energy :np.ndarray,Q :np.ndarray) -> np.ndarray:
    A_omega = np.zeros_like(omega)
    Q0 = Q[np.shape(Q)[0] // 2 ]

    for q_i in range(np.shape(Q)[0]):
        A_omega[:] +=  delta(Energy(Exciton_Energy,Q[q_i])-Energy(Exciton_Energy,Q0)-omega,sigma)
    return A_omega/np.shape(Q)[0]

def DOS_ref(n_ref :int,Δ :np.ndarray,Exciton_Energy :np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    Q_ref,Q_red_ref,Q0_ref,Q0_red_ref = create_Q(n_ref,Δ)

    E_min, E_max = 1.0, 0.0
    Energies = np.zeros((np.shape(Q_red_ref)[0]))

    for qi in range(np.shape(Q_red_ref)[0]):
        E_val = Energy(Exciton_Energy,Q_red_ref[qi])
        Energies[qi] = E_val
        if E_val >=E_max:
            E_max = E_val
        if E_val <=E_min:
            E_min = E_val

    DOS_grid      = np.linspace(E_min-10*(E_max-E_min)/n_ref,E_max+10*(E_max-E_min)/n_ref,n_ref)
    cuentas       = np.zeros((n_ref-1))
    center_energy = np.zeros((n_ref-1))
    Δ_ω           = np.zeros((n_ref-1))
    for e_i in range(1,n_ref):
        Δ_ω[e_i-1] =  DOS_grid[e_i] - DOS_grid[e_i-1]

        if e_i<np.shape(DOS_grid)[0]-1:
            cuentas[e_i-1] = np.sum( (Energies>=DOS_grid[e_i-1]) & (Energies<DOS_grid[e_i]) )*1/Δ_ω[e_i-1]
        else:
            cuentas[e_i-1] = np.sum( (Energies>=DOS_grid[e_i-1]) & (Energies<=DOS_grid[e_i]) )*1/Δ_ω[e_i-1]

        center_energy[e_i-1] = (DOS_grid[e_i-1] + DOS_grid[e_i])/2

    #cuentas*=Δ_ω#/n_ref**len(Δ)
    return center_energy,cuentas,Δ_ω

#============================================================================
#####----------------------------Matrix Γ------------------------------------
def ω_phonon(velocity :np.ndarray,q_vector :np.ndarray) -> float:
    return np.sqrt(np.einsum('i,i', velocity**2, q_vector**2))

def nbose(beta :float,w :float) -> float:
    return 1.0 / (np.exp(beta*w) - 1.0 + 1.0e-6)

def Energy(Exciton_Energy :np.ndarray,q_vector :np.ndarray) -> float:
    return Exciton_Energy[0] + np.einsum('i,i', Exciton_Energy[1:], q_vector**2)

def dExdEydEz(Exciton_Energy :np.ndarray,q_vector :np.ndarray) -> Tuple[float,float,float]:
    return 2*Exciton_Energy[1]*q_vector[0]+1.0e-6,2*Exciton_Energy[2]*q_vector[1]+1.0e-6,2*Exciton_Energy[3]*q_vector[2]+1.0e-6


def delta(E,sigma,Δ_ω):
    #return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2)
    #return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2) * 1/(n**len(Δ)*normalizacion)
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2)*Δ_ω #* 1/(n**len(Δ)*normalizacion)

def Gamma_Q_Qp(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy,Q,Qp):
    β = 1. / (cst.kB * Temp)

    return 2.0*np.pi*( \
                   nbose( β,ω_phonon(velocity,Q-Qp) )       *
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) + ω_phonon(velocity,Q-Qp) ,sigma,Δ_ω) +
                 ( nbose( β,ω_phonon(velocity,Q-Qp) ) +1.0 )*
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) - ω_phonon(velocity,Q-Qp) ,sigma,Δ_ω)
                       )


#def Gamma_Q_Qp(sigma, Temp, velocity, Exciton_Energy, Q, Qp):
def Gamma_Q_Qp_1(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy,Q,Qp):
    β = 1. / (cst.kB * Temp)

    return 2.0*np.pi*( \
                   nbose( β,ω_phonon(velocity,Q-Qp) )       *
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) + ω_phonon(velocity,Q-Qp) ,sigma,Δ_ω)
                       )

def Gamma_Q_Qp_2(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy,Q,Qp):
    β = 1. / (cst.kB * Temp)

    return 2.0*np.pi*( \
                 ( nbose( β,ω_phonon(velocity,Q-Qp) ) +1.0 )*
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) - ω_phonon(velocity,Q-Qp) ,sigma,Δ_ω)
                       )

# Parallelized Gamma_Matrix using OpenMP and MPI

def Gamma_Matrix_worker(qi,n,Δ,Δ_ω,sigma, Temp, velocity, Exciton_Energy, Q):

    γ_prime_row = np.zeros(len(Q))
    for qj in range(len(Q)):
        Q_flips = flipping(Q[qj])
        γ_prime = 0.0
        #-----------------------------------------------------------------
        #np.vectorize(Gamma_Q_Qp)(a, b[:, np.newaxis])
        for q_flips in Q_flips:
            #if not np.array_equal(Q[qj], Q_flips[0]):
            γ_prime += np.linalg.norm(Q[qi] - Q_flips[q_flips] )  * \
                    Gamma_Q_Qp(n,Δ,Δ_ω,sigma, Temp, velocity, Exciton_Energy, Q[qi], q_flips)

        γ_prime_row[qj] = γ_prime
    return γ_prime_row

#def Gamma_Matrix(dimension, sigma, Temp, velocity, Exciton_Energy, Q):
def Gamma_Matrix(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy, Q ):
    dimension = (2*n-1)**len(Δ)
    #Γ_prime = Parallel(n_jobs=3)(delayed(Gamma_Matrix_worker)(qi, sigma, Temp, velocity, Exciton_Energy, Q) for qi in range(len(Q)))
    Γ_prime = Parallel(n_jobs=3)(delayed(Gamma_Matrix_worker)(qi, n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy,Q) for qi in range(len(Q)))
    #return np.array(Γ_prime) / dimension
    return np.array(Γ_prime)

#============================================================================
#####-------------------------System of ODE----------------------------------
def system(t,P, Γ_prime,α,envelope):
    dP = np.zeros_like(P)

    source_term = α/np.sqrt(2*np.pi*envelope**2)*np.exp( -(t-4.5*envelope)**2 /(2.0*envelope**2) )

    dP = np.einsum('ji,j->i',Γ_prime.T,P) - np.einsum('i,i->i',P,np.einsum('ij->i',Γ_prime))
    dP[0] += source_term

    #dP = np.einsum('ji,jt->it',Γ_prime.T,P) - np.einsum('it,i->it',P,np.einsum('ij->i',Γ_prime))
    #dP[0,:] += source_term

    return dP

#============================================================================
#####---------------------Main function to run calculations------------------
def run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,DOS=None,reading_gamma=None,save_gamma=False):
    print("................................")
    print("........Reading Data............")
    print("................................\n")

    n = Q_parameters['n']
    Δ = Q_parameters['Δ']
    σ = Q_parameters['σ']
    Q,Q_red,Q0,Q0_red = create_Q(n,Δ)

    #quit()
    print(f"Points per dimension: {n}")
    print(f"Number of Dimensions: {len(Δ)}")
    print(f"Total number of Points: {np.shape(Q)[0]}")
    print(f"Reduced number of Points: {np.shape(Q_red)[0]}")
    print(f"Δq: {Δ} (au)")

    velocity       = Γ_parameters['velocity_phonons']
    Exciton_Energy = Γ_parameters['Exciton_Energy_Contants']
    Temp           = Γ_parameters['Temp']

    print(f"Phonon Velocity: {velocity} (au)")
    print(f"Excitonic Energy Constants: {Exciton_Energy} (au)")
    print(f"Temperature: {Temp} (K)\n")

    #if DOS_parameters != None:
    print("................................")
    print("......Computing Optimal σ.......")
    print("................................\n")

    n_ref=81
    center_energy,cuentas,Δ_ω = DOS_ref(n_ref,Δ,Exciton_Energy)

    ω_resolution = np.zeros((len(Δ)))
    for eje in range(0,len(Δ)):
        Q_aux,Q_red_aux,Q0_aux,Q0_red_aux = create_Q(n,np.array([Δ[eje]]) )
        Energies_aux = np.zeros((np.shape(Q_red_aux)[0]))

        for qi in range(np.shape(Q_red_aux)[0]):
            E_val = Energy( np.array([ Exciton_Energy[0],Exciton_Energy[eje+1] ]),Q_red_aux[qi])
            Energies_aux[qi] = E_val

        Energies_aux_sorted = np.sort(Energies_aux)
        differences = np.diff(Energies_aux_sorted)
        ω_resolution[eje] = np.min(differences)

    min_difference = np.mean(ω_resolution)

    sigma = σ*min_difference

    print(f"Energy Resolution Δω: {min_difference} (au)")
    print(f"Linear Constand k: {σ}")
    print(f"Optimal σ=kΔω: {sigma} (au)\n")

    #-------------------------------------------------------------------------------------------------------
    A_omega = np.zeros((np.shape(center_energy)[0]))


    for q_i in range(np.shape(Q_red)[0]):
        A_omega[:] +=  delta( Energy(Exciton_Energy,Q_red[q_i])-center_energy,sigma,1.0 )#*1/Δ_ω[e_i-1]

    Q_magnitud        = np.sqrt(np.sum(Q_red**2,axis=1))
    Q_magnitud_sorted = np.sort(Q_magnitud)
    Δ_Q               = Q_magnitud_sorted[1] - Q_magnitud_sorted[0]

    σ_Q = np.zeros((np.shape(Q_red)[0]))
    A_adaptative = np.zeros((np.shape(center_energy)[0]))
    for q_i in range(np.shape(Q_red)[0]):
        dEx, dEy, dEz = dExdEydEz(Exciton_Energy,Q_red[q_i])
        σ_Q[q_i] = 1.0  *np.sqrt( dEx**2 + dEy**2 + dEz**2 )*Δ_Q
        A_adaptative[:] +=  delta( Energy(Exciton_Energy,Q_red[q_i])-center_energy,σ_Q[q_i],1.0 )#*1/Δ_ω[e_i-1]
    #quit()
    #-------------------------------------------------------------------------------------------------------
    if DOS != None:
        print("................................")
        print("........Generating DOS..........")
        print("................................\n")

        plt.figure()
        plt.title(f"n = {n}")
        plt.plot(center_energy*cst.Ry,cuentas/n_ref**len(Δ),color='black',label="Reference")
        plt.plot(center_energy*cst.Ry,A_omega/n**len(Δ),'--',color="red",label=f"$\\Delta\\omega = {min_difference*cst.Ry:.3e} (eV)$")
        plt.plot(center_energy*cst.Ry,A_adaptative/n**len(Δ),'--',color="blue",label=f"$\\sigma_Q = a |\\frac{{\\partial}}{{\\partial Q}} \\epsilon(Q)  | \\Delta Q$")
        plt.legend(fontsize=12)
        plt.xlabel('Energy (eV)',fontsize=12)
        plt.ylabel(f'DOS (arbitrary units)',fontsize=12)
        plt.tick_params(axis="x", labelsize=12)
        plt.tick_params(axis="y", labelsize=12)
        plt.savefig('update/' + f"DOS_n={n}.png")
        #plt.show()

        print("--------------DONE--------------\n")

    print("................................")
    print(".........Computing Γ............")
    print("................................\n")
    if reading_gamma==None:
        start_time  = time.time()
        #Γ_prime = Gamma_Matrix((2*n-1)**len(Δ),sigma,Temp,velocity,Exciton_Energy, Q_red )
        Γ_prime = Gamma_Matrix(n,Δ,1.0,sigma,Temp,velocity,Exciton_Energy, Q_red ) /np.shape(Q_red)[0]
        finish_time = time.time()
        print(f"Computational Time: {finish_time-start_time} \n")

    else:
        print(f'You have imported the Γ-matrix: {reading_gamma}')
        print(f'Reading...')
        Γ_prime = np.loadtxt(reading_gamma)/np.shape(Q_red)[0]
        print(f"File: {reading_gamma}")
    #quit()
    print("--------------DONE--------------\n")
    if save_gamma==True:
        print(f'You have created the file: Gamma_Matrix_paralel_n={n}.txt')
        saving_path = './../../../DATA/data_Boltzmann_equation/Gamma_matrix/'
        np.savetxt(saving_path + f'Gamma_Matrix_paralel_n={n}.txt',Γ_prime*np.shape(Q_red)[0])

    print("................................")
    print(".........Solving ODE............")
    print("................................\n")

    t_init   = time_parameters['t_init']
    t_final  = time_parameters['t_final']
    t_points = time_parameters['t_points']
    source   = time_parameters['source']

    t_span   = (t_init, t_final)
    t_eval   = np.linspace(t_init, t_final, t_points)


    print(f"Initial Time $t_0$: {t_init} (fs)")
    print(f"Final Time $t_{{f}}$: {t_final} (fs)")
    print(f"Time Resolution: {t_points}")

    if source ==True:
        switch='on'
        FWHM      = time_parameters['FWHM']
        amplitude = time_parameters['amplitude']
        envelope  = FWHM/(2.0*np.sqrt(2*np.log(2)))

        Q0     *= 0.0
        Q0_red *= 0.0

        print(f"FWHM: {FWHM} (au)")
        print(f"amplitude: {amplitude} ")

    else:
        switch='off'
        envelope  = 1.0
        amplitude = 0.0

    γ     = γ_parameters['γ']

    print(f"γ constant: {γ}")

    Γ_corrected = Γ_prime

    #γ = γ**2
    for γ_i in γ:
        solution = solve_ivp(system, t_span, Q0_red,t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10, vectorized=False,args=(γ_i*Γ_corrected,amplitude,envelope,))


        """gaussian = amplitude/np.sqrt(2*np.pi*envelope**2)*np.exp( -(t_eval-4.5*envelope)**2 /(2.0*envelope**2) )
        integral = np.zeros_like(t_eval)
        for tt in range(t_points):
            #rd_x[tt] = integrate.trapezoid(velocity_x,tiempo_parcial)
            integral[tt] = np.trapz(gaussian[0:tt],t_eval[0:tt])

        plt.figure()
        plt.plot(t_eval,integral)
        plt.plot(t_eval,solution.y[0],'--')
        plt.axhline(amplitude)
        plt.show()
        quit()"""

        soluciones=np.zeros((np.shape(Q0_red)[0],len(t_eval)))
        for i in range(0,np.shape(Q0_red)[0]):
            y_solution = solution.y[i]
            soluciones[i,:]= y_solution
        path = './../../DATA/data_Boltzmann_equation/populations/'

        print("................................")
        if source ==True:
            hf = h5py.File(path+f'Boltzmann_Evolution_nq={n}_γ={γ_i}_laser={switch}_FWHM={FWHM*cst.tfs:.2f}_α={amplitude:.3f}.h5', 'w')
            hf.create_dataset('E(t)'  , data = 1/np.sqrt(2*np.pi*envelope**2)*np.exp( -(t_eval-4.5*envelope)**2 /(2.0*envelope**2) )          )


            print(f"File created: ./Boltzmann_Evolution_nq={n}_γ={γ_i}_laser={switch}_FWHM={FWHM*cst.tfs:.2f}_α={amplitude:.3f}.h5")

        else:
            hf = h5py.File(path+f'Boltzmann_Evolution_nq={n}_γ={γ_i}_laser={switch}.h5', 'w')
            print(f"File created: ./Boltzmann_Evolution_nq={n}_γ={γ_i}_laser_{switch}.h5")
        hf.create_dataset('Q_grid'  , data = Q_red          )
        hf.create_dataset('P_q(t)'  , data = soluciones     )
        hf.create_dataset('time'    , data = t_eval*cst.tfs )
        hf.close()

        print(f"In folder: {path}")
        print("................................\n")

        nsnap = 4
        step = len(t_eval) // nsnap
        dist = solution.y[:,::step]

        A_omega = np.zeros((dist.shape[-1],np.shape(center_energy)[0]))
        for istep in range(dist.shape[-1]):
            for q_i in range(np.shape(Q_red)[0]):
                A_omega[istep,:] +=  dist[q_i,istep]*delta( Energy(Exciton_Energy,Q_red[q_i])-center_energy,sigma,1 )
                #A_omega[istep,:] +=  delta( Energy(Exciton_Energy,Q[q_i])-center_energy,sigma,Δ_ω[0] )
            #A_omega[istep,:] *= 1/n**len(Δ)
            #normalizacion = np.sum(A_omega[istep,:])
            #A_omega[istep,:] *= 1/normalizacion

        fig = plt.figure(figsize=(8, 13))
        #fig.suptitle(f'Number of points: {n}, Electron-Phonon Scatter rate $\\gamma$: {γ_i}', fontsize=16)
        # Create a GridSpec with 3 rows and 2 columns
        gs = fig.add_gridspec(4, 2)

        # Create the array-like structure for subplots
        ax = [[None, None], [None, None], [None, None],  [None, None]]
        ax[0][0] = fig.add_subplot(gs[0, :])  # Row 0, spanning both columns
        for i in range(0,np.shape(Q0_red)[0]):

            ax[0][0].plot(t_eval * cst.tfs, soluciones[i,:])
        ax[0][0].set_xlabel('Time (fs)',fontsize=12)
        ax[0][0].set_ylabel(f'$\\rho(t)$',fontsize=12)
        ax[0][0].tick_params(axis="x", labelsize=12)
        ax[0][0].tick_params(axis="y", labelsize=12)
        #ax[0][0].set_ylim(0,0.020)

        ax[0][0].grid(True)
        #ax[0][0].legend()
        ax[0][0].set_title(f'Number of points: {n}, Electron-Phonon Scatter rate $\\gamma$: {γ_i}', fontsize=14)
        # Assign None to ax[0][1] to maintain the array structure consistency
        ax[0][1] = ax[0][0]  # Keep it the same to maintain consistency but won't be used directly

        cmap = plt.get_cmap('tab10')  # You can choose different colormaps like 'viridis', 'plasma', etc.
        colors = [cmap(i) for i in range(nsnap)]


        ax[1][0] = fig.add_subplot(gs[1, :])  # Row 0, spanning both columns

        for istep in range(dist.shape[-1]):
            print(trapezoid(A_omega[istep,:],center_energy))
            #ax[1][0].plot(center_energy*cst.Ry,A_omega[istep,:]/np.shape(Q_red)[0]/np.max(A_omega[istep,:]/np.shape(Q_red)[0]),'--',color=colors[istep],label=f"t = {t_eval[istep*step]* cst.tfs:.3f} fs")
            ax[1][0].plot(center_energy*cst.Ry,A_omega[istep,:]/np.shape(Q_red)[0] ,'--',color=colors[istep],label=f"t = {t_eval[istep*step]* cst.tfs:.3f} fs")
        ax[1][0].set_xlabel('Energy (eV)',fontsize=12)
        ax[1][0].set_ylabel(f'$A({{\\omega}},t)$ (arbitrary units)',fontsize=12)
        ax[1][0].tick_params(axis="x", labelsize=12)
        ax[1][0].tick_params(axis="y", labelsize=12)
        #ax[1][0].set_ylim(0,0.00030)

        ax[1][0].legend(fontsize=12)
        ax[0][1] = ax[0][0]

        iteraciones_plot = [i for i in range(2, int(nsnap/2) + 2)]
        second_component = [0, 1]

        pairs = list(itertools.product(iteraciones_plot, second_component))

        for istep in range(dist.shape[-1]):
            #ax[1][0] = fig.add_subplot(gs[1, 0])  # Second row, first column
            ax[pairs[istep][0]][pairs[istep][1]] = fig.add_subplot(gs[ pairs[istep][0], pairs[istep][1] ])  # Second row, first column
            ax[pairs[istep][0]][pairs[istep][1]].plot(np.sort(dist[:,istep])[::-1],'.',color=colors[istep],label=f"t = {t_eval[istep*step]* cst.tfs:.3f} fs")
            ax[pairs[istep][0]][pairs[istep][1]].plot(np.sort(dist[:,istep])[::-1],':',color=colors[istep])
            #ax[pairs[istep][0]][pairs[istep][1]].set_ylim(-0.001,0.01)
            ax[pairs[istep][0]][pairs[istep][1]].legend(fontsize=12)

            if pairs[istep][0] <iteraciones_plot[-1]:
                ax[pairs[istep][0]][pairs[istep][1]].set_xticklabels([])
            else:
                ax[pairs[istep][0]][pairs[istep][1]].set_xlabel('Points',fontsize=12)
            if pairs[istep][1] ==0:
                ax[pairs[istep][0]][pairs[istep][1]].set_ylabel(f'$\\rho(t)$',fontsize=12)
            else:
                ax[pairs[istep][0]][pairs[istep][1]].set_yticklabels([])
            ax[pairs[istep][0]][pairs[istep][1]].tick_params(axis="x", labelsize=12)
            ax[pairs[istep][0]][pairs[istep][1]].tick_params(axis="y", labelsize=12)
            #ax[pairs[istep][0]][pairs[istep][1]].set_ylim(0,0.020)

            ax[0][0].axvline(x=t_eval[istep*step]* cst.tfs,color=colors[istep], ls='--')

        #plt.show()
        if source ==True:
            plt.savefig('update/' + f"Population_q_independent_n={n}_gamma={γ_i}_laser_{switch}_FWHM={FWHM*cst.tfs:.2f}_α={amplitude:.3f}.png", bbox_inches='tight', transparent=False)
        else:
            plt.savefig('update/' + f"Population_q_independent_n={n}_gamma={γ_i}_laser_{switch}.png", bbox_inches='tight', transparent=False)

        print("################################")
        print("...........Finished.............")
        print("################################\n")
    return 0

def main():

    Q_parameters = {
        'n': 21,
        'Δ': np.array([0.035,0.035,0.035]),
        'σ': 8.652109
        #'σ': 5.9715
                    }

    Γ_parameters = {
        'velocity_phonons'        : np.array([2.102e-3,4.390e-3,2.330e-3]),
        'Exciton_Energy_Contants' : np.array([0.01066335,0.22180669,0.58639659,0.08477976]),
        'Temp'                    : 80
                    }

    time_parameters = {
            't_init'    : 0.0,
            't_final'   : 7000 / cst.tfs,
            't_points'  : 20000,
            'source'    : True,
            'FWHM'      : 300 / cst.tfs,
            'amplitude' : 0.016
                        }

    DOS_parameters = {
            'ω_min' : -0.01/cst.Ry,
            'ω_max' :  0.05/cst.Ry,
            'ω_n'   :  2001
                      }

    #γ = np.array([0.00005,0.00006,0.00007,0.00008,0.00009,0.0001])
    γ_parameters = {
        #'γ' : np.array([0.000001,0.0000009,0.0000007,0.0000005,0.0000003,0.0000001,0.00000009,0.00000007,0.00000005,0.00000003])
        #'γ' : np.array([0.00,0.000000001,0.000001])
        #'γ' : np.array([0.00000001])
        #'γ' : np.array([0.000000002,0.000000003,0.000000001])
        #'γ' : np.array([0.000001,0.0000001,0.00])
        'γ' : np.array([0.000001,0.0000001])
        #'γ' : np.array([0.00000001,0.000000001,0.0])
                   }

    #run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,DOS=True,reading_gamma=None,save_gamma=True)
    run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,DOS=True,reading_gamma=f"./../../DATA/data_Boltzmann_equation/Gamma_matrix/Gamma_reduced_n={Q_parameters['n']}.txt",save_gamma=False)



#------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

