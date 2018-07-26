import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding

import numpy as np
from scipy.integrate import odeint

from gym.envs.temp_ctrl.Models import sysParam as sysParam
# import Models as Models

'''param = ['Vaccan', 'Seism']
act_space = ['D10', 'D20', 'D50', 'D100', 'D200', 'D500', 'C']
reward_type = ['Rw10', 'Rw4', 'Rquad', 'Rexp']
ambtemp_models = ['Tcon', 'Tsin', 'Trand', 'Tsinrand']
timestep_size = ['t1', 't10', 't30', 't60', 't100']'''

class TempCtrlEnvs(gym.Env):
    metadata = {
        'render.modes':['human', 'rgb_array']
    }

    def __init__(self,
                 thermalParam='Vaccan',
                 act_space='D200',
                 reward_type='Rexp',
                 ambtemp_models='Tsin',
                 timestep_size='t10'):

        #  Configure system thermal params or throw error if unknown
        if thermalParam == 'Vaccan':
#            self.k = 1.136*25.e-3
#            self.m = 15.76
#            self.C = 505
#            self.A = 1.3
#            self.d = 5.08e-2

            self.k, self.m, self.C, self.A, self.d = sysParam.VacCanParams()
        elif thermalParam == 'Seism':
            self.k, self.m, self.C, self.A, self.d = sysParam.SeismParams()
        else:
            raise ValueError(
                'Thermal parameter specifier not in known list of systems.')

        self.t_step = 0.1  # seconds between state updates

        if timestep_size[0] is 't':
            self.t_max = int(timestep_size[1:])  # 10 seconds = 1 time-step
        else:
            raise ValueError(
                'Error: timestep_size must start with t, specifier bad')

        # configure discreet or cont action space or throw error of unknown
        if act_space in ['D10', 'D20', 'D50', 'D100', 'D200', 'D500']:
            sizeActionSpace = int(act_space[1:])  # conv to useable number
            self.action_space = spaces.Discrete(float(sizeActionSpace))
        elif act_space is 'C':
            self.action_space = spaces.Box(np.array([0.]),
                                           np.array([100.]),
                                           dtype=np.float64)
        else:
            raise ValueError('Error: unknown act_space specifier.')

        self.observation_space = spaces.Box(np.array([15.0, 0.0]),
                                            np.array([60.0, 50.0]),
                                            dtype=np.float64)
        # initial seed and reset of env
        self.elapsed_steps = 0
        self.seed()
        self.reset()

    # Sets seed for random number generator used in the environment
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = [self.np_random.uniform(low=15, high=30), self.T_amb(0)]
        self.steps_beyond_done = None
        return np.array(self.state)

    # todo: this is hardcoded here, reimplement external Model.py import
    def vac_can(self, T, t_inst):
            dTdt = -self.k*self.A*(T-self.T_amb(t_inst))/(self.d*self.m*self.C)+self.P_heat/(self.m*self.C)
            return dTdt

    # todo: this is hardcoded in for now until decorator is written
    def T_amb(self, time):
        """Returns ambient temperature oscillating around 20 C with an
           amplitude of 5 C, depending on number of steps elapsed. """
        return 5*np.sin(2*np.pi*(self.elapsed_steps*10. + time)/(6*3600)) + 20.

    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.elapsed_steps += 1
        T_can = self.state[0]
        self.P_heat = action

        # todo: do you really need to allocate on each step here? Can just do
        # in __init__?
        self.t = np.arange(0, self.t_max, self.t_step)

        #  gets final value after integration
        T_can_updated = float(odeint(
            self.vac_can, T_can, self.t)[int(self.t_max/self.t_step) - 1])

        self.state = np.array([T_can_updated,
                               self.T_amb(self.elapsed_steps*10.)])

        done = T_can_updated < 15. or T_can_updated > 60. # kill run if railed
        done = bool(done)

        # todo: hard codeing reward, need to refactor rewards as class
        #reward = Models.Rewar
        T_setpoint = 45
        if not done:
            if T_can_updated > T_setpoint-5. and T_can_updated <= T_setpoint+5.:
                reward = 0.1
            else:
                reward = 0.

        return self.state, reward, done, {}
