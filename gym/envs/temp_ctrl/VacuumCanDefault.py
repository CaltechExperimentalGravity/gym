import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint


#####################################################
################## Vac Can Discrete #################
#####################################################
class VacCanEnvDiscrete(gym.Env):
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
        self.t_max = 10  # 10 seconds = 1 time-step

        # Set-point temperature
        self.T_setpoint = 45  # Celsius

        # Temperature at which to fail the episode
        self.T_threshold = 60

        self.action_space = spaces.Discrete(35)
        self.observation_space = spaces.Box(np.array([15.0, 0.0]),
                                            np.array([60.0, 50.0]),
                                            dtype=np.float64)

        self.seed()
        # self.state = None
        self.steps_beyond_done = None
        self.elapsed_steps = 0
        self.reset()


# Sets seed for random number generator used in the environment
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# Physical Model of Vacuum Can temperature
    def vac_can(self, T, t_inst):
        # dTdt = -self.k*self.A*(T-self.T__env_buff[np.argmax(self.t >=\
        #        t_inst)])/(self.d*self.m*self.C) \
        #       + self.H_buff[np.argmax(self.t>= t_inst)]/(self.m*self.C)

        dTdt = -self.k*self.A*(T-self.T_amb(t_inst))/(self.d*self.m*self.C) \
               + self.P_heat/(self.m*self.C)
        return dTdt

# Ambient temperature function/list
    def T_amb(self, time):
        """Returns ambient temperature oscillating around 20 C with an
           amplitude of 5 C, depending on number of steps elapsed. """
        return 5*np.sin(2*np.pi*(time)/(6*3600)) + 20.


# Simulates reaction
    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.elapsed_steps += 1

        T_can = self.state[0]

        self.P_heat = action

        self.t = np.arange(0, self.t_max, self.t_step) + self.elapsed_steps*10.

        #  self.T__env_buff = np.interp(self.t, self.t, T_amb)
        #  self.H_buff = np.interp(self.t, self.t, P_heat)

        #  gets final value after integration
        T_can_updated = float(odeint(
            self.vac_can, T_can, self.t)[int(self.t_max/self.t_step) - 1])

        self.state = np.array([T_can_updated,
                               self.T_amb(self.elapsed_steps*10.)])

        done = T_can_updated < 15. or T_can_updated > 60.
        done = bool(done)


        if not done:
            if self.state[0] > 40. and self.state[0] < 50.:
                reward = 0.1
            else:
                reward = 0.
        elif self.steps_beyond_done is None:

            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this "
                            "environment has already returned done = True. "
                            "You should always call 'reset()' once you "
                            "receive 'done = True' -- any further steps are "
                            "undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.state = [self.np_random.uniform(low=15, high=30), self.T_amb(0)]
        self.steps_beyond_done = None
        return np.array(self.state)

#####################################################
################## Vac Can Continuous ###############
#####################################################
class VacCanEnvContinuous(gym.Env):
    metadata = {
        'render.modes':['human']
    }

    def __init__(self):
        self.k = 1.136*25e-3   # comments go here !
        self.m = 15.76
        self.C = 505
        self.A = 1.3
        self.d = 5.08e-2
        self.t_step = 0.1      # seconds between state updates
        self.t_max = 10        # 10 seconds = 1 time-step

        # Set-point temperature
        self.T_setpoint = 45  # Celsius

        # Temperature at which to fail the episode
        self.T_threshold = 90

        self.action_space = spaces.Box(np.array([0.]),
                                       np.array([100.]), dtype=np.float64)
        self.observation_space = spaces.Box(np.array([15.0, 0.0]),
                                            np.array([60.0, 50.0]),
                                            dtype=np.float64)

        self.seed()
        # self.state = None
        self.steps_beyond_done = None
        self.elapsed_steps = 0
        self.reset()


# Sets seed for random number generator used in the environment
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# Physical Model of Vacuum Can temperature
    def vac_can(self, T, t_inst):
        # dTdt = -self.k*self.A*(T-self.T__env_buff[np.argmax(self.t >=\
        #        t_inst)])/(self.d*self.m*self.C) \
        #       + self.H_buff[np.argmax(self.t>= t_inst)]/(self.m*self.C)

        dTdt = -self.k*self.A*(T-self.T_amb(t_inst))/(self.d*self.m*self.C) \
               + self.P_heat/(self.m*self.C)
        return dTdt

# Ambient temperature function/list
    def T_amb(self, time):
        """Returns ambient temperature oscillating around 20 C with an
           amplitude of 5 C, depending on number of steps elapsed. """
        return 5*np.sin(2*np.pi*(self.elapsed_steps*10.)/(24*3600)) + 20.


    # Simulates reaction
    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.elapsed_steps += 1

        T_can = self.state[0]

        self.P_heat = action

        self.t = np.arange(0, self.t_max, self.t_step)

        #  self.T__env_buff = np.interp(self.t, self.t, T_amb)
        #  self.H_buff = np.interp(self.t, self.t, P_heat)

        #  gets final value after integration
        T_can_updated = float(odeint(
            self.vac_can, T_can, self.t)[int(self.t_max/self.t_step) - 1])

        self.state = np.array([T_can_updated,
                               self.T_amb(self.elapsed_steps*10.)])

        done = T_can_updated < 15. or T_can_updated > 60.
        done = bool(done)


        if not done:
            if self.state[0] > 40. and self.state[0] < 50.:
                reward = 0.1
            else:
                reward = 0.
        elif self.steps_beyond_done is None:

            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this "
                            "environment has already returned done = True. "
                            "You should always call 'reset()' once you "
                            "receive 'done = True' -- any further steps are "
                            "undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.state = [self.np_random.uniform(low=15, high=30), self.T_amb(0)]
        self.steps_beyond_done = None
        return np.array(self.state)
