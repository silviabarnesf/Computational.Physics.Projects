A mass-spring system undergoes damped harmonic motion with mass m = 1.0 kg; the spring constant is k = 1.0 N/m, and the damping coefficient c = 0.1 kg/s. The system starts
with an initial displacement of x₀ = 1.0 m and an initial velocity of v₀ = 0.0 m/s. 

The goals are: 1. Solve the differential equations (derive the equation of motion for the damped harmonic oscillator & using numerical methods, solve the second-order differential 
equation to determine x(t) and v(t) over time); 2. Use a root-finding algorithm to determine the first time at which the displacement x(t) = 0 (i.e., when the object crosses the equilibrium position);
3. Compute the Fourier transform of the displacement x(t) over time.


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import solve_ivp

# Constants
m = 1.0  # mass (kg)
k = 1.0  # spring constant (N/m)
c = 0.1  # damping coefficient (kg/s)

# Define the system of equations
def damped_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -0.1 * v - x  # Equation for acceleration
    return [dxdt, dvdt]

# Initial conditions
x0 = 1.0  # initial displacement (m)
v0 = 0.0  # initial velocity (m/s)
y0 = [x0, v0]  # Initial state

# Time range for the solution
t_span = (0, 10)  # From 0 to 10 seconds
t_eval = np.linspace(0, 10, 1000)  # Time points at which to evaluate the solution

# Solve the differential equations using solve_ivp
solution = solve_ivp(damped_oscillator, t_span, y0, t_eval=t_eval)

# Extract the displacement and velocity from the solution
x_sol = solution.y[0]
v_sol = solution.y[1]

# Plot displacement vs. time
plt.figure(figsize=(10, 6))
plt.plot(solution.t, x_sol, label='Displacement x(t)', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Displacement vs. Time for Damped Harmonic Oscillator')
plt.grid(True)
plt.legend()
plt.show()

# Plot velocity vs. time
plt.figure(figsize=(10, 6))
plt.plot(solution.t, v_sol, label='Velocity v(t)', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs. Time for Damped Harmonic Oscillator')
plt.grid(True)
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import fft, fftfreq

# Constants
m = 1.0  # Mass (kg)
k = 1.0  # Spring constant (N/m)
c = 0.1  # Damping coefficient (kg/s)

# Initial conditions
x0 = 1.0  # Initial displacement (m)
v0 = 0.0  # Initial velocity (m/s)

# Time span
t_max = 50  # Duration of simulation (seconds)
dt = 0.01   # Time step
t_values = np.arange(0, t_max, dt)  # Time array

# Equation of motion: dx/dt = v, dv/dt = (-k/m)x - (c/m)v
def damped_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (-k/m) * x - (c/m) * v
    return [dxdt, dvdt]

# Solve the differential equation
sol = solve_ivp(damped_oscillator, [0, t_max], [x0, v0], t_eval=t_values)
x_values = sol.y[0]  # Extract displacement x(t)

# Compute FFT
X_f = fft(x_values)  # Compute FFT of x(t)
freqs = fftfreq(len(t_values), d=dt)  # Frequency axis
positive_freqs = freqs[freqs > 0]  # Positive half of frequencies
positive_X_f = np.abs(X_f[freqs > 0])  # Magnitude spectrum

# Theoretical frequencies
f_natural = (1 / (2 * np.pi)) * np.sqrt(k / m)  # Undamped natural frequency
f_damped = (1 / (2 * np.pi)) * np.sqrt(k/m - (c**2 / (4 * m**2)))  # Damped frequency

# Plot the Fourier Transform
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_X_f, label='Damped Oscillator Spectrum', color='b')
plt.axvline(f_natural, color='g', linestyle='--', label=f'Undamped f = {f_natural:.2f} Hz')
plt.axvline(f_damped, color='r', linestyle='--', label=f'Damped f = {f_damped:.2f} Hz')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Damped Harmonic Oscillator")
plt.grid(True)
plt.legend()
plt.show()

