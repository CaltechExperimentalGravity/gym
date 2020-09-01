# -*- coding: utf-8 -*-
"""
Custom gym environment to train network for temperature control of seismometer.

Uses continuous input and action space.

"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

class TempEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.k = 0.026        # thermal conductivity, W/(m*K)
        self.m = 12.2         # object mass, kg
        self.c = 500          # heat capacity, J/kg*k
        self.A = 2            # area, m^2
        self.d = 0.1          # thickness, m
        self.dt = 1           # time interval, s
        self.t_max = 10       # 10 seconds = 1 time-step
        self.min_heat = 0     # heater off = 0 W
        self.max_heat = 30    # max heater power

        # Set-point temperature
        self.T_setpoint = 35  # Celsius

        #End episode if temperature exceeds max
        self.T_max = 60 # Celsius

        # Discrete spaces
        # set action space as discrete {0,20}?
        # self.action_space = spaces.Discrete(20)
        
        # Continuous spaces
        self.action_space = spaces.Box(
            np.array([0.]),
            np.array([100.]),
            dtype=np.float64
        )
        
        self.observation_space = spaces.Box(
            np.array([15.0, 0.0]),np.array([60.0, 50.0]),
            dtype=np.float64
        )

        self.seed()
        self.elapsed_steps = 0
        self.reset

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # defining the plant
    def model(self, T, t_inst): #for ODE-int solver

        #Varying T_amb
        #dTdt = -self.k * A_foam / (m_ss * c_ss * h_foam) * (const.convert_temperature(T_amb(t),'Celsius','Kelvin') - T)
        print("Heating Power = {} W".format(self.P_in))
        # Constant T_amb
        dTdt = -self.k * self.A / (self.m * self.c * self.d) * (T - self.T_amb(t_inst)) + self.P_in/(self.m * self.c)
        return dTdt

    # generate ambient temperature
    def T_amb(self, time):
        # for now just a constant temperature
        return 0*time + 20 #Celsius

    # system response, and take action (supply heater power)
    def step(self, action):
        '''

        self.P_in = action #heating
        T_obj, dTdt = self.state
        dTdt += (action - 1) / (self.m*self.c)
        T_obj += dTdt

        done = bool(
            T_obj = self.T_setpoint)
        reward = -1.0

        if not done:
            reward = -abs(self.state[0]-self.T_setpoint)

            if self.state[0] > 30. and self.state[0] < 40.:
                reward = 0.2
            elif self.state[0] > 50. or self.state[0] < 20.:
                reward = -0.2
            else:
                reward = 0.

        self.state = (T_obj, dTdt)


        '''
        """assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))"""


        self.elapsed_steps += 1

        T_obj     = self.state[0]
        self.P_in = action # heating
        u = np.clip(self.P_in, -self.min_heat, self.max_heat)
        self.P_in = u
        self.last_u = u  # for rendering

        self.t = np.arange(0, self.t_max, self.dt) + self.elapsed_steps * 10

        #  gets final value after integration
        T_new = float(odeint(self.model, T_obj, self.t)[int(self.t_max/self.dt) - 1])
        #T_new = float(odeint(
        #    self.model(T_obj, self.elapsed_steps*10),T_obj,self.t)[int(self.t_max/self.dt) - 1])

        # state contains object temp & ambient temp
        self.state = np.array([T_new, self.T_amb(self.elapsed_steps * 10)])

        # finish if temperature goes out of these bounds
        done = T_new < 1 or T_new > 90
        done = bool(done)

        if not done:
            reward = -abs(self.state[0] - self.T_setpoint)

            # use setpoint here rather than hard-coded values
            #if self.state[0] > 30 and self.state[0] < 40:
            #    reward = 0.2
            #elif self.state[0] > 50 or self.state[0] < 20:
            #    reward = -0.2
            #else:
            #    reward = 0



        return self.state, reward, done, {}

    # reset the environment
    def reset(self):
        #setting bounds for new starting temperature
        high = 25
        low = 15
        T_start = self.np_random.uniform(low=low, high=high)
        #self.state = [T_start, self.model(T_start,0)]
        self.state = [T_start, self.T_amb(0)]
        return np.array(self.state)


    def render(self, mode='human'):
        #store time series temperature
        templist = []
        templist.append(self.state[0])
        print('Temperature =  {:0.3f} C'.format(self.state[0]))
        #print('Supplied power: ',{self.P_in})
