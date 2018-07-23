import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

                ##### Parameters in model for thermal dynamics #####
def VacCanParams():
    k = 1.136*25e-3   # Thermal conductivity constant of foam
    m = 15.76   # Mass of vacuum can
    C = 505    # Specific heat capacity of vac can
    A = 1.3   # Cross-sectional area normal to conduction
    d = 5.08e-2   # Thickness of foam

def SeismParams(): #!!!!!! Change based on Kira's elogs
    k = 1.136*25e-3   # Thermal conductivity constant of foam
    m = 15.76   # Mass of vacuum can
    C = 505    # Specific heat capacity of vac can
    A = 1.3   # Cross-sectional area normal to conduction
    d = 5.08e-2   # Thickness of foam


#            ###### Heat conduction equation ######
def ModelEquation(self, T, t_inst):

    dTdt = -self.k*self.A*(T-self.T_amb(t_inst))/(self.d*self.m*self.C) \
           + self.P_heat/(self.m*self.C)
    return dTdt

            ###### Ambient temperature models #####
Tamb_standard = 20.

def TambConstant(t):
    return Tamb_standard

def TambRandom(t):
    return np.random.random()*5. + Tamb_standard

def TambSine(elapsed_steps):
    time_period = 24. # hours
    amplitude = 5. # degrees Celsius
    return amplitude*np.sin(2*np.pi*elapsed_steps*timestep)/(time_period*3600) + Tamb_standard


            ###### Reward functions ######
def RewardWindow10(T_can_updated, T_setpoint):
    if T_can_updated > T_setpoint-5. and T_can_updated <= T_setpoint+5.:
        reward = 0.1
    else:
        reward = 0.
    return reward

def RewardWindow4(T_can_updated):
    if T_can_updated > T_setpoint-2. and T_can_updated <= T_setpoint+2.:
        reward = 0.1
    else:
        reward = 0.
    return reward

def RewardExp(T_can_updated, T_setpoint):
    return np.exp(-(T_can_updated-T_setpoint)**2/(2*T_setpoint))

def RewardQuadratic(T_can_updated, T_setpoint):
    return 1. - (T_can_updated - T_setpoint)**2/T_setpoint**2


        ###### Reset state ######
def reset():
    state = [np_random.uniform(low=15, high=30), self.T_amb(0)]
    self.steps_beyond_done = None
    return np.array(self.state)
