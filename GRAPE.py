import numpy as np
import scipy.linalg
from scipy.linalg import kron
from scipy.optimize import minimize

# Define constants
N = 7  # Number of spins in the chain
J = 1  # Coupling strength
dt = 0.15  # Time step
hbar = 1  # Reduced Planck's constant (setting to 1 for simplicity)
B_on = 100  # Magnetic field strength when on
fidelity_threshold = 0.99  # Desired fidelity
noise_level = 0.1  # Noise level as a fraction of B_on

# Define Hamiltonian components
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

def pauli_z():
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)

def identity():
    return np.eye(2, dtype=np.complex128)

def create_hamiltonian(J, B_fields):
    N = len(B_fields)
    H = np.zeros((2 ** N, 2 ** N), dtype=np.complex128)

    for n in range(N - 1):
        H_xx = np.eye(1, dtype=np.complex128)
        H_yy = np.eye(1, dtype=np.complex128)
        for k in range(N):
            if k == n or k == n + 1:
                H_xx = kron(H_xx, pauli_x())
                H_yy = kron(H_yy, pauli_y())
            else:
                H_xx = kron(H_xx, identity())
                H_yy = kron(H_yy, identity())
        H += (J / 2) * (H_xx + H_yy)

    for n in range(N):
        H_z = np.eye(1, dtype=np.complex128)
        for k in range(N):
            if k == n:
                H_z = kron(H_z, pauli_z())
            else:
                H_z = kron(H_z, identity())
        H += B_fields[n] * H_z

    return H

def time_evolution(psi, H, dt):
    U = scipy.linalg.expm(-1j * H * dt)
    return U @ psi

def fidelity(psi1, psi2):
    return np.abs(np.dot(np.conj(psi1), psi2)) ** 2

# Define target state
target_psi = np.random.rand(2 ** N) + 1j * np.random.rand(2 ** N)
target_psi /= np.linalg.norm(target_psi)

# Noise model: add Gaussian noise to the magnetic fields
def add_noise(B_fields, noise_level):
    noise = np.random.normal(0, noise_level, size=B_fields.shape)
    return B_fields + noise

# GRAPE optimization
def grape_optimization(target_function, x_start, max_iter=1000):
    result = minimize(lambda B_fields: -target_function(B_fields), x_start, method='L-BFGS-B',
                      bounds=[(0, 1)] * len(x_start), options={'maxiter': max_iter})
    return result.x, -result.fun

# Target function to maximize fidelity
def target_function(B_fields):
    B_fields = np.array(B_fields) * B_on  # Convert binary fields to B_on
    B_fields_noisy = add_noise(B_fields, noise_level * B_on)  # Add noise to the magnetic fields

    psi0 = np.random.rand(2 ** N) + 1j * np.random.rand(2 ** N)
    psi0 /= np.linalg.norm(psi0)

    H = create_hamiltonian(J, B_fields_noisy)  # Use noisy magnetic fields in Hamiltonian
    psi_t = time_evolution(psi0, H, dt)
    psi_t /= np.linalg.norm(psi_t)

    return fidelity(psi_t, target_psi)

# Initial guess for magnetic fields
x_start = np.zeros(N)

# Loop until the fidelity threshold is achieved
max_iter = 1000  # Maximum iterations for the GRAPE algorithm in each loop
total_max_iter = 5000  # Total maximum iterations allowed
optimal_fidelity = 0
iteration_count = 0
episode = 0
highest_fidelity = 0

while optimal_fidelity < fidelity_threshold and iteration_count < total_max_iter:
    episode += 1
    result_B_fields, result_fidelity = grape_optimization(target_function, x_start, max_iter=max_iter)
    optimal_B_fields = result_B_fields * B_on  # Convert binary fields to B_on
    optimal_fidelity = result_fidelity
    iteration_count += max_iter
    if optimal_fidelity > highest_fidelity:
        highest_fidelity = optimal_fidelity
    print(f"Episode {episode}: Current Optimal Fidelity: {optimal_fidelity}")

    if iteration_count >= total_max_iter:
        break

print("Optimal magnetic fields:", optimal_B_fields)
print("Optimal fidelity:", highest_fidelity)
print("Total iterations:", iteration_count)
