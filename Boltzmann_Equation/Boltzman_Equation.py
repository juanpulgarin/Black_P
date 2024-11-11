import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint
import itertools

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
def create_Q(n,Δ):
    dimensiones = len(Δ)

    if dimensiones == 1:
        Q = create_Q_1D(n,Δ)

        Q0=np.zeros(np.shape(Q)[0])
        Q0[n**1 // 2] = 1.0
        return Q,Q0
    if dimensiones == 2:
        Q = create_Q_2D(n,Δ)

        Q0=np.zeros(np.shape(Q)[0])
        Q0[n**2 // 2] = 1.0
        return Q,Q0
    if dimensiones == 3:
        Q = create_Q_3D(n,Δ)

        Q0=np.zeros(np.shape(Q)[0])
        Q0[n**3 // 2] = 1.0
        return Q,Q0

def create_Q_3D(n,Δ):
    x = np.linspace(-Δ[0],Δ[0],n)
    y = np.linspace(-Δ[1],Δ[1],n)
    z = np.linspace(-Δ[2],Δ[2],n)
    xv,yv,zv = np.meshgrid(x,y,z)

    Q = np.zeros((n**3,3))
    for i in range(n**3):
        Q[i,0] = xv.reshape(n**3)[i]
        Q[i,1] = yv.reshape(n**3)[i]
        Q[i,2] = zv.reshape(n**3)[i]

    return Q

def create_Q_2D(n,Δ):
    x = np.linspace(-Δ[0],Δ[0],n)
    y = np.linspace(-Δ[1],Δ[1],n)
    xv,yv = np.meshgrid(x,y)

    Q = np.zeros((n**2,2))
    for i in range(n**2):
        Q[i,0] = xv.reshape(n**2)[i]
        Q[i,1] = yv.reshape(n**2)[i]

    return Q

def create_Q_1D(n,Δ):
    x = np.linspace(-Δ[0],Δ[0],n)

    Q = np.zeros((n**1,1))
    for i in range(n**1):
        Q[i,0] = x[i]

    return Q

#============================================================================
#####-----------------------Density of States--------------------------------
def DOS(sigma,omega,Exciton_Energy,Q):
    A_omega = np.zeros_like(omega)
    Q0 = Q[np.shape(Q)[0] // 2 ]

    for q_i in range(np.shape(Q)[0]):
        A_omega[:] +=  delta(Energy(Exciton_Energy,Q[q_i])-Energy(Exciton_Energy,Q0)-omega,sigma)
    return A_omega/np.shape(Q)[0]

#============================================================================
#####----------------------------Matrix Γ------------------------------------
def ω_phonon(velocity,q_vector):
    return np.sqrt( np.einsum('i,i',velocity**2,q_vector**2) )

def nbose(beta,w):
    return 1.0 / ( np.exp(beta*w) -1.0 )

def Energy(Exciton_Energy,q_vector):
    return Exciton_Energy[0] + np.einsum('i,i',Exciton_Energy[1:],q_vector**2)
    #return Exciton_Energy[0] + Exciton_Energy[1]*q_vector[0]**2 + Exciton_Energy[2]*q_vector[1]**2 + Exciton_Energy[3]*q_vector[2]**2

def delta(E,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*E**2/sigma**2)

def Gamma_Q_Qp(sigma,Temp,velocity,Exciton_Energy,Q,Qp):
    #print(np.shape(Q),np.shape(Qp),np.shape(velocity),np.shape(Exciton_Energy))
    β=1. / (cst.kB * Temp)

    return 2.0*np.pi*( \
                   nbose( β,ω_phonon(velocity,Q-Qp) )       *
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) + ω_phonon(velocity,Q-Qp) ,sigma) +
                 ( nbose( β,ω_phonon(velocity,Q-Qp) ) +1.0 )*
                         delta( Energy(Exciton_Energy,Q) - Energy(Exciton_Energy,Qp) - ω_phonon(velocity,Q-Qp) ,sigma)
                       )

def Gamma_Matrix(sigma,Temp,velocity,Exciton_Energy, Q ):
    Γ = np.zeros((np.shape(Q)[0],np.shape(Q)[0]))

    for qi in range(np.shape(Q)[0]):
        for qj in range(qi):
            if qi != qj:
                gamma = Gamma_Q_Qp(sigma,Temp,velocity,Exciton_Energy, Q[qi], Q[qj] )
                Γ[                     qi,                     qj] = gamma
                Γ[(np.shape(Q)[0]-1) - qi,(np.shape(Q)[0]-1) - qj] = gamma

    return Γ/np.shape(Q)[0]

#============================================================================
#####-------------------------System of ODE----------------------------------
def system(t,P, Γ):
    dP = np.zeros_like(P)

    dP = np.einsum('ji,jt->it',Γ,P) - np.einsum('it,i->it',P,np.einsum('ij->i',Γ))

    return dP

#============================================================================


def run_calc(time_parameters,Γ_parameters, γ, Q_parameters,DOS_parameters=None):
    n = Q_parameters['n']
    Δ = Q_parameters['Δ']
    σ = Q_parameters['σ']
    Q,Q0 = create_Q(n,Δ)



    #quit()
    velocity       = Γ_parameters['velocity_phonons']
    Exciton_Energy = Γ_parameters['Exciton_Energy_Contants']
    Temp           = Γ_parameters['Temp']

    if DOS_parameters != None:
        omega = np.linspace(DOS_parameters['ω_min'],DOS_parameters['ω_max'],DOS_parameters['ω_n'])

        A_omega = DOS(σ,omega,Exciton_Energy,Q)
        np.savetxt('Large_DOS.txt',(omega,A_omega))
        plt.figure()
        plt.plot(omega*cst.Ry,A_omega)
        plt.show()

    #quit()
    Γ = Gamma_Matrix(σ,Temp,velocity,Exciton_Energy, Q )

    t_init   = time_parameters['t_init']
    t_final  = time_parameters['t_final']
    t_points = time_parameters['t_points']
    t_span = (t_init, t_final)
    t_eval = np.linspace(t_init, t_final, t_points)

    for γ_i in γ:
        solution = solve_ivp(system, t_span, Q0,t_eval=t_eval, vectorized=True,args=(γ_i**2*Γ,))


        #plt.figure()
        #nsnap = 5
        #step = len(t_eval) // nsnap
        #print(step)

        #dist = solution.y[:,::step]
        #for istep in range(dist.shape[-1]):
        #    plt.scatter(Q, dist[:,istep],label=t_eval[istep*step]* cst.tfs)

        #plt.legend()


        soluciones=np.zeros((np.shape(Q0)[0],len(t_eval)))
        plt.figure()
        for i in range(0,np.shape(Q0)[0]):
        #for i in range(10,17):

            y_solution = solution.y[i]
            soluciones[i,:]= y_solution

            plt.plot(t_eval * cst.tfs,y_solution,'-',label = Q[i])

        #plt.legend()
        plt.xlabel('t (fs)')
        plt.ylabel(r'y(t)')
        plt.title(f'{γ_i}')
        plt.ylim(0,0.2)

        plt.grid(True)
        plt.show()

        combined_array = np.vstack((t_eval, soluciones))
        np.savetxt('Boltzman_Complete.txt',(combined_array))

    return 0


#----------------------------------------------------------------------
"""def main():

    Q_parameters = {
        'n': 3,
        'Δ': np.array([0.025,0.025,0.025]),
        'σ': 0.0001
                    }

    Γ_parameters = {
        'velocity_phonons'        : np.array([2.102e-3,4.390e-3,2.330e-3]),
        'Exciton_Energy_Contants' : np.array([0.01066335,0.22180669,0.58639659,0.08477976]),
        'Temp'                    : 80
                    }

    time_parameters = {
            't_init'   : 0.0,
            't_final'  : 200 / cst.tfs,
            't_points' : 500
                        }

    DOS_parameters = {
            'ω_min' : -0.1/cst.Ry,
            'ω_max' :  0.15/cst.Ry,
            'ω_n'   :  2001
                      }

    γ = np.array([.00005])

    run_calc(time_parameters,Γ_parameters, γ, Q_parameters)""";

def main():

    Q_parameters = {
        'n': 3,
        'Δ': np.array([0.025,0.025,0.025]),
        'σ': 0.003
                    }

    Γ_parameters = {
        'velocity_phonons'        : np.array([2.102e-3,4.390e-3,2.330e-3]),
        'Exciton_Energy_Contants' : np.array([0.01066335,0.22180669,0.58639659,0.08477976]),
        'Temp'                    : 80
                    }

    time_parameters = {
            't_init'   : 0.0,
            't_final'  : 2500 / cst.tfs,
            't_points' : 2500
                        }

    DOS_parameters = {
            'ω_min' : -0.01/cst.Ry,
            'ω_max' :  0.05/cst.Ry,
            'ω_n'   :  2001
                      }

    γ = np.array([0.0001])

    run_calc(time_parameters,Γ_parameters, γ, Q_parameters)




#----------------------------------------------------------------------
if __name__ == '__main__':
    main()

