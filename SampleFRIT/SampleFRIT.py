import numpy as np
from scipy.signal import lti, dlsim, dlti
import csv

# Simulation of FIRT

# Initial settings
Ts = 0.01
A = np.array([[1, 1], [0, -2]])
B = np.array([[0], [1]])
x0 = np.array([0, 0])
n = A.shape[0]
F_ini = np.array([-0.8, 2.0])
Hd = [
    dlti([1], [1, -0.5, 0], dt=Ts),
    dlti([1, -1], [1, -0.5, 0], dt=Ts)
]

# Simulation
k = np.arange(0, 51)  # Step count
N = len(k)
t = np.arange(0, N*Ts, Ts)  # Time vector

# Rectangular wave signal
v = np.zeros(N)
v[2:7] = 1

# State update
x = np.zeros((N, 2))
x[0, :] = x0
for index in range(1, N):
    u = np.dot(F_ini, x[index-1, :]) + v[index-1]
    temp = np.dot(A, x[index-1, :]) + B.flatten() * u
    x[index, :] = temp

# Initial data
x_ini = x
u_ini = np.dot(F_ini, x_ini.T) + v

# Minimize objective function
Gamma = np.zeros(n * N)
W = np.zeros((n * N, n))
for j in range(n):
    # Using lsim for discrete-time systems
    _, y_out1, _ = dlsim(Hd[j], u_ini, t)
    Gamma[j * N:(j + 1) * N] = x_ini[:, j] - y_out1.flatten()

    _, y_out2, _ = dlsim(Hd[j], x_ini[:, 0], t)
    _, y_out3, _ = dlsim(Hd[j], x_ini[:, 1], t)
    W[j * N:(j + 1) * N, :] = np.column_stack((y_out2.flatten(), y_out3.flatten()))

# Write Gamma, W, W'*W to .csv files
np.savetxt('Gamma.csv', Gamma, delimiter=',')
np.savetxt('W.csv', W, delimiter=',')
np.savetxt('Psi.csv', np.dot(W.T, W), delimiter=',')

# Calculate optimal state feedback gain
F_ast = -np.dot(np.dot(Gamma.T, W), np.linalg.inv(np.dot(W.T, W)))

# Display results
print('F_ast =')
print(F_ast)
