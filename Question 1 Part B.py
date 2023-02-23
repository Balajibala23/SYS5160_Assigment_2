from cmath import e
import numpy as np
from scipy.integrate import odeint
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define the universe of discourse for the error signal e(t) and the control signal V(t)
e = ctrl.Antecedent(np.arange(-4, 4.1, 0.1), 'e')
v = ctrl.Consequent(np.arange(0, 50.1, 0.1), 'v')

# Define the fuzzy sets and membership functions for e(t) and V(t)
e['L'] = fuzz.trimf(e.universe, [-4, -4, -1])
e['M'] = fuzz.trimf(e.universe, [-2, 0, 2])
e['H'] = fuzz.trimf(e.universe, [1, 4, 4])
v['L'] = fuzz.trimf(v.universe, [0, 0, 16])
v['M'] = fuzz.trimf(v.universe, [12, 25, 38])
v['H'] = fuzz.trimf(v.universe, [32, 50, 50])

# Define the fuzzy rules for the controller
rule1 = ctrl.Rule(e['L'], v['L'])
rule2 = ctrl.Rule(e['M'], v['M'])
rule3 = ctrl.Rule(e['H'], v['H'])

# Create the fuzzy logic controller and add the rules
ctrl_sys = ctrl.ControlSystem([rule1, rule2, rule3])
ctrl_sim = ctrl.ControlSystemSimulation(ctrl_sys)

# Define the system parameters
R = 2
b = 0.05
#a = 0.2
Vmax = 48
H0 = 1
d = 4
Hd = 5  # Desired height

# Define the function that describes the dynamics of the water tank system
def water_tank(h, t, v):
    if (t<=20):
        dhdt = ((b * v) - 0.1 * np.sqrt(h)) / (np.pi * R**2)
    elif(t>20):
        dhdt = ((b * v) - 2 * 0.2 * np.sqrt(h)) / (np.pi * R**2)
    return dhdt

# Define the time points for simulation
t = np.linspace(0, 1000, 10000)
h_list = []

# Simulate the system using the fuzzy logic controller
h = H0
for i in range(len(t)-1):
    e = Hd - h
    ctrl_sim.input['e'] = e
    ctrl_sim.compute()
    v = ctrl_sim.output['v']
    v = min(v, Vmax)  # Limit the voltage to the maximum value
    h = odeint(water_tank, h, [t[i], t[i+1]], args=(v,))[1][0]
    
    if h > Hd:
        h = Hd
    h_list.append(h)
    

# Plot the results
plt.plot(t, np.ones_like(t)*Hd, 'k--', label='Desired height')
plt.plot(t[:-1], h_list, 'b', linewidth=2, label='Actual height')
plt.legend(loc='best')
plt.xlabel('Time (min)')
plt.ylabel('Height (m)')
plt.ylim([0, 10])
plt.show()