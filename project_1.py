Build up the same matrix using at least 7 different ways, including Monte Carlo, Gaussian quadratic, Trapezoidal and Simpson Integration methods; and quantum computing logic gates based on 1-qubit basis states.

import numpy as np
import scipy.integrate as spi
from scipy.integrate import quad

# Build 1: Quadratic Integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
I_quad, error = quad (lambda x: np.real (f(x)), A, B)
#print(f"quad: {I_quad}, Error: {error}")

I_quad, error = quad (lambda x: np.imag (f(x)), A, B)
#print(f"quad: {I_quad}, Error: {error}")

bell_state = 1/np.sqrt(2)*np.array(([1],[0],[0],[1]))
#print(bell_state*I_quad)


#Build 1.2: Quadratic integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
I_quad, error = quad (lambda x: np.real (f(x)), A, B)
#print(f"quad: {I_quad}, Error: {error}")

I_quad, error = quad (lambda x: np.imag (f(x)), A, B)
#print(f"quad: {I_quad}, Error: {error}")

state_00 = np.array(([1],[0],[0],[0]))
state_11 = np.array(([0],[0],[0],[1]))
bell_state = 1/np.sqrt(2)*(state_00+state_11)
#print(bell_state)

print(bell_state*I_quad)


# Build 2: Trapezoidal Integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
x_vals = np.linspace(A,B,1200)
y_vals = np.real(f(x_vals))
I_trapz = np.trapezoid(y_vals,x_vals)
#print(f"Trapezoidal rule: {I_trapz}")

x_vals = np.linspace(A,B,1200)
y_vals = np.imag(f(x_vals))
I_trapz = np.trapezoid(y_vals,x_vals)
#print(f"Trapezoidal rule: {I_trapz}")

bell_state = 1/np.sqrt(2)*np.array(([1],[0],[0],[1]))
print(bell_state*I_trapz)


#Build 2.2: Trapezoidal Integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
x_vals = np.linspace(A,B,1200)
y_vals = np.real(f(x_vals))
I_trapz = np.trapezoid(y_vals,x_vals)
#print(f"Trapezoidal rule: {I_trapz}")

x_vals = np.linspace(A,B,1200)
y_vals = np.imag(f(x_vals))
I_trapz = np.trapezoid(y_vals,x_vals)
#print(f"Trapezoidal rule: {I_trapz}")

state_00 = np.array(([1],[0],[0],[0]))
state_11 = np.array(([0],[0],[0],[1]))
bell_state = 1/np.sqrt(2)*(state_00+state_11)
#print(bell_state)

#print(bell_state*I_trapz)


# Build 3: Simpsons Integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
x_vals = np.linspace(A,B,1200)
y_vals = np.real(f(x_vals))
I_simps = spi.simpson(y_vals, x_vals)
#print(f"Simpson's rule: {I_simps}")

x_vals = np.linspace(A,B,1200)
y_vals = np.imag(f(x_vals))
I_simps = spi.simpson(y_vals, x_vals)
#print(f"Simpson's rule: {I_simps}")

x_vals = np.linspace(A,B,1200)
y_vals = f(x_vals)
I_simps = spi.simpson(y_vals, x_vals)
#print(f"Simpson's rule: {I_simps}")

bell_state = 1/np.sqrt(2)*np.array(([1],[0],[0],[1]))
print(bell_state*I_simps)



#Build 3.2: Simpson's Integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
x_vals = np.linspace(A,B,1200)
y_vals = np.real(f(x_vals))
I_simps = spi.simpson(y_vals, x_vals)
#print(f"Simpson's rule: {I_simps}")

x_vals = np.linspace(A,B,1200)
y_vals = np.imag(f(x_vals))
I_simps = spi.simpson(y_vals, x_vals)
#print(f"Simpson's rule: {I_simps}")

x_vals = np.linspace(A,B,1200)
y_vals = f(x_vals)
I_simps = spi.simpson(y_vals, x_vals)
#print(f"Simpson's rule: {I_simps}")

state_00 = np.array(([1],[0],[0],[0]))
state_11 = np.array(([0],[0],[0],[1]))
bell_state = 1/np.sqrt(2)*(state_00+state_11)
#print(bell_state)

#print(bell_state*I_simps)


#Build 4: Monte Carlo Integration
A= 0
B= np.pi/2
def f(x):
    return 1/(1+np.cos(x)) * np.exp(-3j*x) * np.sin(2*x)
def monte_carlo(f, n):
    x_vals = np.random.uniform(A,B,n)
    y_vals = f(x_vals)
    return np.sum(y_vals)

I_mc = monte_carlo(f, 1000000)
print(f"Monte Carlo: {I_mc}")
