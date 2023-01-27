import qutip
import numpy as np

from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.qip.operations import hadamard_transform
import qutip.logging_utils as logging
from qutip.control.grape import cy_grape_unitary, grape_unitary_adaptive, plot_grape_control_fields, _overlap
from qutip.ui.progressbar import TextProgressBar

dtype = np.complex64
X = np.array([[0, 1], [1, 0]],dtype=dtype)
Y = np.array([[0, -1j], [1j, 0]],dtype=dtype)
Z = np.array([[1, 0], [0, -1]],dtype=dtype)
I = np.eye(2,dtype=dtype)

def construct_hamiltonian(nqubit,dtype=np.complex64):
    ## this function return a tuple 
    ## where the first element is the drift hamiltonian
    ## the second element is a list of control hamiltonian    
    Hc = np.zeros([2**nqubit,2**nqubit],dtype= dtype)
    Hd = []
    ZZ = np.kron(Z,Z)
    for i in range(nqubit-1):
        Hc += np.kron(np.kron(np.eye(2**i),ZZ),np.eye(2**(nqubit-i-2)))
        Hd.append(np.kron(np.kron(np.eye(2**i),X),np.eye(2**(nqubit-i-1))))
        Hd.append(np.kron(np.kron(np.eye(2**i),Y),np.eye(2**(nqubit-i-1))))
    Hd.append(np.kron(np.eye(2**(nqubit-1)),X))
    Hd.append(np.kron(np.eye(2**(nqubit-1)),Y))
    return Hc,Hd

def to_qobj(H,nqubit):
    dims = [ [2 for i in range(nqubit)] for i in range(2)]
    return Qobj(H[0],dims=dims),[Qobj(tmp,dims=dims) for tmp in H[1]]

dt = 0.05
N_list = np.array([20,80,300,1200,5000])
## we use 100 iterations to estimate a single iteration
R = 50
for i,nqubit in enumerate(range(2,7)):
    print("---"*5+"\n"+"benchmark for %d qubits \t number of iterations %d"%(nqubit,R))
    H_c, H_d = to_qobj(construct_hamiltonian(nqubit),nqubit)
    times = np.linspace(0,N_list[i]*dt,N_list[i])
    U_targ = hadamard_transform(nqubit)
    u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.01 for _ in range(len(H_d))])
    result = str(cy_grape_unitary(U_targ, H_c, H_d, R, times, phase_sensitive=False,
                          u_start=u0, progress_bar=TextProgressBar(),
                          eps=2*np.pi*5))
    print(result)
