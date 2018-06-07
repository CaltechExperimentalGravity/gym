import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint


class VacCanEnv(gym.Env):
    metadata = {
        'render.modes':['human']
    }

    def __init__(self):
        self.k = 1.136*25e-3
        self.m = 15.76
        self.C = 505
        self.A = 1.3
        self.d = 5.08e-2
        self.t_step = 0.1  # seconds between state updates
        self.t_max = 10*10 # 10 seconds = 1 time-step

        # Set-point temperature
        self.T_setpoint = 45  # Celsius

        # Temperature at which to fail the episode
        self.T_threshold = 60

        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(np.array([15.0]), np.array([60.0]))

        self.seed()
        # self.state = None
        self.reset()
        self.steps_beyond_done = None


# Sets seed for random number generator used in the environment
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# Physical Model of Vacuum Can temperature
    def vac_can(self, T, t_inst):
        # dTdt = -self.k*self.A*(T-self.T__env_buff[np.argmax(self.t >= t_inst)])/(self.d*self.m*self.C) \
        #       + self.H_buff[np.argmax(self.t>= t_inst)]/(self.m*self.C)
               
        dTdt = -self.k*self.A*(T-self.T_amb)/(self.d*self.m*self.C) + self.P_heat/(self.m*self.C)
        return dTdt


# Simulates reaction
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        T_can = self.state

        self.P_heat = action*20
        self.T_amb = 5*np.sin(2*np.pi*self.t_step/(6*3600)) + 20  # Ambient temperature oscillating around 20 C with an amplitude of 5 C

        self.t = np.arange(0, self.t_max, self.t_step)
        
        #self.T__env_buff = np.interp(self.t, self.t, T_amb)
        #self.H_buff = np.interp(self.t, self.t, P_heat)

        T_can_updated = odeint(self.vac_can, T_can[0], self.t)[int(self.t_max/self.t_step) -1]

        self.state = T_can_updated

        done = T_can_updated[0] < 15 or T_can_updated[0] > 60
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:

            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this"
                            "environment has already returned done = True."
                            "You should always call 'reset()' once you receive"
                            " 'done = True' -- any further steps are "
                            "undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=15, high=30)])
        self.steps_beyond_done = None
        return np.array(self.state)
