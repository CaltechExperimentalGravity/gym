import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding

import numpy as np
from scipy.integrate import odeint

#import gym.envs.temp_ctrl.Models as Models
#import models

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
            self.k, self.m, self.C, self.A, self.d = Models.VacCanParams()
        elif thermalParam == 'Seism':
            self.k, self.m, self.C, self.A, self.d = Models.SeismParams()
        else:
            raise ValueError(
                'Thermal parameter specifier not in known list of systems.')
        self.t_step = 0.1  # seconds between state updates
        self.t_max = int(timestep_size[1:])  # 10 seconds = 1 time-step

        # configure discreet or continuous action space
        if act_space in ['D10', 'D20', 'D50', 'D100', 'D200', 'D500', 'C']:
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

        self.seed()
        self.steps_beyond_done = None
        self.elapsed_steps = 0
        self.reset()


    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.elapsed_steps += 1
        T_can = self.state[0]
        self.P_heat = action

        self.t = np.arange(0, self.t_max, self.t_step)

        #  gets final value after integration
        T_can_updated = float(odeint(
            self.vac_can, T_can, self.t)[int(self.t_max/self.t_step) - 1])

        self.state = np.array([T_can_updated,
                               self.T_amb(self.elapsed_steps*10.)])

        done = T_can_updated < 15. or T_can_updated > 60.
        done = bool(done)

        reward = Models.Rewar

        return self.state, reward, done, {}
