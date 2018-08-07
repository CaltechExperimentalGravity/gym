import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

                ##### Parameters in model for thermal dynamics #####
class sysParam():
    def VacCanParams():
        k = 1.136*25e-3   # Thermal conductivity constant of foam
        m = 15.76   # Mass of vacuum can
        C = 505    # Specific heat capacity of vac can
        A = 1.3   # Cross-sectional area normal to conduction
        d = 5.08e-2   # Thickness of foam
        return k, m, C, A, d

    def SeismParams(): #!!!!!! Change based on Kira's elogs
        k = 1.136*25e-3   # Thermal conductivity constant of foam
        m = 15.76   # Mass of vacuum can
        C = 505    # Specific heat capacity of vac can
        A = 1.3   # Cross-sectional area normal to conduction
        d = 5.08e-2   # Thickness of foam
        return k, m, C, A, d

                ###### Ambient temperature models #####
class TambModels():
    Tamb_standard = 20.

    def TConstant():
    return Tamb_standard

    def TRandom():
    return np.random.random()*5. + Tamb_standard

    def TSine(elapsed_steps, time, timestep):
    time_period = 24. # hours
    amplitude = 5. # degrees Celsius

    return amplitude*np.sin(2*np.pi*elapsed_steps*timestep + time)/(time_period*3600) + Tamb_standard

    def TSineRandom(elapsed_steps, time, timestep):
    time_period = 24. # hours
    amplitude = 5. # degrees Celsius

    return amplitude*np.sin(2*np.pi*elapsed_steps*timestep + time)/(time_period*3600)/2 + np.random.random()*amplitude/2 + Tamb_standard



            ###### Reward functions ######
class rewardType():
    def RewardWindow10(T_can_updated, T_setpoint):
        if T_can_updated > T_setpoint-5. and T_can_updated <= T_setpoint+5.:
            reward = 0.1
        else:
            reward = 0.
        return reward

    def RewardWindow4(T_can_updated, T_setpoint):
        if T_can_updated > T_setpoint-2. and T_can_updated <= T_setpoint+2.:
            reward = 0.1
        else:
            reward = 0.
        return reward

    def RewardExp(T_can_updated, T_setpoint):
        return 0.1*np.exp(-(T_can_updated-T_setpoint)**2/(2*T_setpoint))

    def RewardQuadratic(T_can_updated, T_setpoint):
        return 0.1*(1. - (T_can_updated - T_setpoint)**2/T_setpoint**2)

    def RewardReciprocalQuadratic(T_can_updated, T_setpoint):
        return 0.1/((T_can_updated - T_setpoint)**2/T_setpoint**2)


#            ###### Heat conduction equation ######
def ModelEquation(self, T, t_inst):

    dTdt = -self.k*self.A*(T-self.T_amb(t_inst))/(self.d*self.m*self.C) \
           + self.P_heat/(self.m*self.C)
    return dTdt
