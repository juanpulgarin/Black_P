import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint,trapezoid
from scipy import interpolate
import itertools
import time
from typing import Tuple

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
    Q0_reduced[0]           = 1.0*0.01
    Q0[n**dimensiones // 2] = 1.0*0.01

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

def flipping(vector):
    sign_options = [[x, -x] if x != 0 else [x] for x in vector]

    combinations = list(itertools.product(*sign_options))
    return  [np.array(comb) for comb in combinations]
"""
def flipping(vector):
    num_nonzero = np.sum(vector != 0)
    num_combinations = 2 ** num_nonzero  # Number of sign combinations for non-zero elements
    combinations = []

    for i in range(num_combinations):
        comb = np.copy(vector)
        flip_index = 0
        for j in range(len(vector)):
            if vector[j] != 0:
                if (i >> flip_index) & 1:  # Flip the sign if bit is set
                    comb[j] = -comb[j]
                flip_index += 1
        combinations.append(comb)

    return combinations"""

#============================================================================
#####--------------------Density of States (old)-----------------------------
def DOS(sigma,omega,Exciton_Energy,Q):
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
def ω_phonon(velocity,q_vector):
    return np.sqrt( np.einsum('i,i',velocity**2,q_vector**2) )

def nbose(beta,w):
    return 1.0 / ( np.exp(beta*w) -1.0 + 1.0e-6)

def Energy(Exciton_Energy,q_vector):
    return Exciton_Energy[0] + np.einsum('i,i',Exciton_Energy[1:],q_vector**2)
    #return Exciton_Energy[0] + Exciton_Energy[1]*q_vector[0]**2 + Exciton_Energy[2]*q_vector[1]**2 + Exciton_Energy[3]*q_vector[2]**2

def dExdEydEz(Exciton_Energy,q_vector):
    return 2*Exciton_Energy[1]*q_vector[0]+1.0e-6,2*Exciton_Energy[2]*q_vector[1]+1.0e-6,2*Exciton_Energy[3]*q_vector[2]+1.0e-6

def delta(E,sigma,Δ_ω):
    #return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2)
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2)*Δ_ω #* 1/(n**len(Δ)*normalizacion)

#def Gamma_Q_Qp(sigma,Temp,velocity,Exciton_Energy,Q,Qp):
def Gamma_Q_Qp(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy,Q,Qp):
    β=1. / (cst.kB * Temp)

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
         nbose(β, ω_phonon(velocity, Q-Qp))*
                delta( Energy(Exciton_Energy, Q) - Energy(Exciton_Energy, Qp) + ω_phonon(velocity, Q-Qp),sigma,Δ_ω  )
                       )

def Gamma_Q_Qp_2(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy,Q,Qp):
    β = 1. / (cst.kB * Temp)

    return 2.0*np.pi*( \
                 ( nbose( β,ω_phonon(velocity,Q-Qp) ) +1.0 )*
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) - ω_phonon(velocity,Q-Qp) ,sigma,Δ_ω)
                       )

def Gamma_Matrix(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy, Q ):
    dimension = (2*n-1)**len(Δ)
    Γ_prime       = np.zeros((np.shape(Q)[0],np.shape(Q)[0]))


    for qi in range(np.shape(Q)[0]):

        for qj in range(np.shape(Q)[0]):
            Q_flips = flipping(Q[qj])
            γ_prime       = 0.0

            for q_flips in range(np.shape(Q_flips)[0]):

                γ_prime  += np.linalg.norm(Q[qi] - Q_flips[q_flips] )* \
                                Gamma_Q_Qp(n,Δ,Δ_ω,sigma,Temp,velocity,Exciton_Energy, Q[qi], Q_flips[q_flips])

            Γ_prime[qi,qj]       = γ_prime

    ##return Γ_prime/(dimension)#,Γ_prime_prime/(dimension)
    return Γ_prime

#============================================================================
#####-------------------------System of ODE----------------------------------
def system(t,P, Γ_prime,α,envelope):
    dP = np.zeros_like(P)

    #dP = np.einsum('ji,jt->it',Γ_prime,P) - np.einsum('it,i->it',P,np.einsum('ij->i',Γ_prime_prime))
    dP = np.einsum('ji,jt->it',Γ_prime,P) - np.einsum('it,i->it',P,np.einsum('ij->i',Γ_prime.T))

    source_term = α*np.exp( -t**2 /(2.0*envelope**2) )

    dp[0,:] += source_term

    return dP

#============================================================================

def run_DOS(time_parameters,Γ_parameters, γ_parameters, Q_parameters,DOS_parameters=None):
    n     = Q_parameters['n']
    Δ     = Q_parameters['Δ']
    σ_K   = Q_parameters['σ']
    Q_run,Q0_run = create_Q(n,Δ)

    velocity       = Γ_parameters['velocity_phonons']
    Exciton_Energy = Γ_parameters['Exciton_Energy_Contants']
    Temp           = Γ_parameters['Temp']


    if DOS_parameters != None:
        n_ref=50
        center_energy,cuentas,Δ_ω = DOS_ref(n_ref,Δ,Exciton_Energy)

        print("................................")
        print(".......Computing DOS............")
        print("................................")

        #--------------------------------------------------------------------------
        ω_resolution = np.zeros((len(Δ)))
        for eje in range(0,len(Δ)):
            Q_aux,Q0_aux = create_Q(n,np.array([Δ[eje]]) )
            Energies_aux = np.zeros((np.shape(Q_aux)[0]))

            for qi in range(np.shape(Q_aux)[0]):
                E_val = Energy( np.array([ Exciton_Energy[0],Exciton_Energy[eje+1] ]),Q_aux[qi])
                Energies_aux[qi] = E_val

            Energies_aux_sorted = np.sort(Energies_aux)
            differences = np.diff(Energies_aux_sorted)
            ω_resolution[eje] = np.min(differences)

        min_difference = np.mean(ω_resolution)
        print(np.mean(ω_resolution))

        sigma=np.linspace(1*min_difference,30*min_difference,10)
        A_omega = np.zeros((np.shape(center_energy)[0],len(sigma)))

        for j,σ in enumerate(sigma):
            for q_i in range(np.shape(Q_run)[0]):
                A_omega[:,j] +=  delta( Energy(Exciton_Energy,Q_run[q_i])-center_energy,σ,1.0 )#*1/Δ_ω[e_i-1]


        #----------------------------------------------------------------------------------------------------
        Q_magnitud        = np.sqrt(np.sum(Q_run**2,axis=1))
        Q_magnitud_sorted = np.sort(Q_magnitud)
        Δ_Q               = Q_magnitud_sorted[1] - Q_magnitud_sorted[0]
        print(Δ_Q)
        σ_Q = np.ones((np.shape(Q_run)[0]))
        for q_i in range(np.shape(Q_run)[0]):
            dEx, dEy, dEz = dExdEydEz(Exciton_Energy,Q_run[q_i])
            σ_Q[q_i] = 1.0  *np.sqrt( dEx**2 + dEy**2 + dEz**2 )*Δ_Q
        print(np.min(σ_Q))


        A_adaptative = np.zeros((np.shape(center_energy)[0]))
        for q_i in range(np.shape(Q_run)[0]):
                A_adaptative[:] +=  delta( Energy(Exciton_Energy,Q_run[q_i])-center_energy,σ_Q[q_i],1.0 )#*1/Δ_ω[e_i-1]

        #----------------------------------------------------------------------------------------------------

        # Create a figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Number of points: {n}, Energy Resolution $\Delta\omega$: {min_difference: .4e} (au)', fontsize=16)


        ax1.plot(center_energy,cuentas/n_ref**len(Δ),'-',color='black',label=f"$\\rho_{{ref}}$")
        ax1.plot(center_energy,A_adaptative/n**len(Δ),'--',color='black',label=f"$σ_{{var}}$")
        for j,σ in enumerate(sigma):
            ax1.plot(center_energy,A_omega[:,j]/n**len(Δ),':',label=f"{σ/min_difference:.3f}")

        ax1.set_xlim(min(center_energy),max(center_energy))
        #ax1.set_ylim(0.0,0.06)
        ax1.legend()
        ax1.set_xlabel("Exciton Energy (au)",fontsize=16)
        ax1.set_ylabel("DOS (arbitrary units)",fontsize=16)
        ax1.tick_params(axis="x", labelsize=16)
        ax1.tick_params(axis="y", labelsize=16)
        ax1.legend(fontsize=12,fancybox=True,framealpha=.7)

        value_fix=np.zeros((np.shape(sigma)[0]))
        for j,σ in enumerate(sigma):
            value_fix[j] = 1/len(center_energy)*sum(abs(A_omega[:,j]/n**len(Δ)-cuentas/n_ref**len(Δ)))
            ax2.plot(σ/min_difference,1/len(center_energy)*sum(abs(A_omega[:,j]/n**len(Δ)-cuentas/n_ref**len(Δ))),'.')#,label=f"{σ/min_difference}")

        f = interpolate.interp1d(sigma,value_fix, kind = 'cubic')
        def interpolated_f(x_val):
            return f(x_val)
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(interpolated_f, bounds=(sigma.min(), sigma.max()), method='bounded')
        min_x = result.x
        min_y = interpolated_f(min_x)

        ax2.plot(min_x/min_difference,min_y,'o',color='black',label=f'K={min_x/min_difference}')

        ax2.legend(fontsize=12,fancybox=True,framealpha=.7)

        #ax2.set_ylim(0.0,0.16)
        ax2.set_xlabel(f"$\sigma = K*\Delta\omega$",fontsize=16)
        ax2.set_ylabel("diff${| \\rho_{{ref}} - \\rho_{{\\sigma}}  |}$",fontsize=16)
        ax2.tick_params(axis="x", labelsize=16)
        ax2.tick_params(axis="y", labelsize=16)

        #plt.savefig(f"Comparison_fitting_sigma_n={n}.png", bbox_inches='tight', transparent=False)

        plt.show()

        return 0

def run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,DOS_parameters=None,reading_gamma=None,save_gamma=False):
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

    n_ref=50
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

    quit()
    print("................................")
    print(".........Computing Γ............")
    print("................................\n")
    if reading_gamma==None:
        start_time  = time.time()
        #Γ_prime = Gamma_Matrix((2*n-1)**len(Δ),sigma,Temp,velocity,Exciton_Energy, Q_red )
        Γ_prime = Gamma_Matrix(n,Δ,1.0,sigma,Temp,velocity,Exciton_Energy, Q_red )
        finish_time = time.time()
        print(f"Computational Time: {finish_time-start_time} \n")

    else:
        print(f'You have imported the Γ-matrix: {reading_gamma}')
        print(f'Reading...')
        Γ_prime = np.loadtxt(reading_gamma)
        print(f"File: {reading_gamma}")
    #quit()
    print("--------------DONE--------------\n")
    if save_gamma==True:
        print(f'You have created the file: Gamma_Matrix_normalized_n={n}.txt')
        np.savetxt(f'Gamma_Matrix_normalized_n={n}.txt',Γ_prime)

    print("................................")
    print(".........Solving ODE............")
    print("................................\n")
    quit()
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
        FWHM      = time_parameters['FWHM']
        amplitude = time_parameters['amplitude']
        envelope  = FWHM/(2.0*np.sqrt(2*np.log(2)))

        print(f"FWHM: {FWHM} (au)")
        print(f"amplitude: {amplitude} ")

    else:
        envelope  = 0.0
        amplitude = 0.0


    γ            = γ_parameters['γ']
    q_dependence = γ_parameters['q_dependence']

    print(f"γ constant: {γ}")
    print(f"is γ q-dependent? {q_dependence} \n")

    if q_dependence == True:
        Γ_corrected = np.zeros_like(Γ_prime)
        for qi in range(np.shape(Q)[0]):

            for qj in range(np.shape(Q)[0]):
                Γ_corrected[qi,qj] = np.linalg.norm(Q[qi]-Q[qj])*Γ_prime[qi,qj]

    else:
        Γ_corrected = Γ_prime

    for γ_i in γ:
        solution = solve_ivp(system, t_span, Q0,t_eval=t_eval, vectorized=True,args=(γ_i**2*Γ_corrected,amplitude,envelope,))


        plt.figure()
        nsnap = 4
        step = len(t_eval) // nsnap
        #print(step)

        dist = solution.y[:,::step]
        for istep in range(dist.shape[-1]):
            plt.plot( dist[:,istep],'.',label=t_eval[istep*step]* cst.tfs)
            #print(sum(dist[:,istep]))
        plt.legend()


        soluciones=np.zeros((np.shape(Q0)[0],len(t_eval)))
        plt.figure()

        for istep in range(dist.shape[-1]):
            plt.axvline(x=t_eval[istep*step]* cst.tfs,c='k', ls='--')
        for i in range(0,np.shape(Q0)[0]):
        #for i in range(10,17):

            y_solution = solution.y[i]
            soluciones[i,:]= y_solution

            plt.plot(t_eval * cst.tfs,y_solution,'-',label = Q[i])




        #plt.legend()
        plt.xlabel('t (fs)')
        plt.ylabel(r'y(t)')
        plt.title(f'{γ_i}')
        plt.ylim(0,0.020)

        plt.grid(True)

        plt.figure()
        A_omega = np.zeros((dist.shape[-1],np.shape(center_energy)[0]))
        for istep in range(dist.shape[-1]):
            for q_i in range(np.shape(Q)[0]):
                A_omega[istep,:] +=  dist[q_i,istep]*delta( Energy(Exciton_Energy,Q[q_i])-center_energy,sigma,Δ_ω[0] )
            A_omega[istep,:] *= 1/n**len(Δ)
            normalizacion = np.sum(A_omega[istep,:])
            A_omega[istep,:] *= 1/normalizacion
            print(sum(A_omega[istep,:]))

            plt.plot(center_energy,A_omega[istep,:])


        plt.savefig(f"test_n_{n}.png")

        combined_array = np.vstack((t_eval, soluciones))
        #np.savetxt('Boltzman_Reduced.txt',(combined_array))


        plt.show()
    return 0


def main():

    Q_parameters = {
        'n': 15,
        'Δ': np.array([0.035,0.035,0.035]),
        'σ': 8.652109
        #'σ': 5.9715
                    }

    Γ_parameters = {
        'velocity_phonons'        : np.array([2.102e-3,4.390e-3,2.330e-3]),
        'Exciton_Energy_Contants' : np.array([0.01066335,0.22180669,0.58639659,0.08477976]),
        'Temp'                    : 80
                    }
    """Q_parameters = {
        'n': 5,
        'Δ': np.array([0.025]),
        'σ': 0.00001
                    }

    Γ_parameters = {
        'velocity_phonons'        : np.array([4.390e-3]),
        'Exciton_Energy_Contants' : np.array([0.01066335,0.58639659]),
        'Temp'                    : 80
                    }"""

    time_parameters = {
            't_init'    : 0.0,
            't_final'   : 2000 / cst.tfs,
            't_points'  : 2000,
            'source'    : True,
            'FWHM'      : 100 / cst.tfs,
            'amplitude' : 1.0
                        }

    DOS_parameters = {
            'ω_min' : -0.01/cst.Ry,
            'ω_max' :  0.05/cst.Ry,
            'ω_n'   :  2001
                      }

    γ_parameters = {
        'γ' : np.array([0.01]),
        'q_dependence' : False
                   }


    #run_DOS(time_parameters,Γ_parameters, γ_parameters, Q_parameters,DOS_parameters=True)
    run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,reading_gamma=None,save_gamma=True)
    #run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,reading_gamma="./Gamma_Matrix_normalized_n=8.txt",save_gamma=False)
    run_calc(time_parameters,Γ_parameters, γ_parameters, Q_parameters,reading_gamma=None,save_gamma=True)




#----------------------------------------------------------------------
if __name__ == '__main__':
    main()

