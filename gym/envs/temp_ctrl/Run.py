import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding

import numpy as np
from scipy.integrate import odeint

from gym.envs.temp_ctrl.Models import sysParam, TambModels, rewardType

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
                 ambtemp_model='Tsin',
                 timestep_size='t10'):

        #  Configure system thermal params or throw error if unknown
        if thermalParam == 'Vaccan':
            self.k, self.m, self.C, self.A, self.d = sysParam.VacCanParams()
        elif thermalParam == 'Seism':
            self.k, self.m, self.C, self.A, self.d = sysParam.SeismParams()
        else:
            raise ValueError(
                'Thermal parameter specifier not in known list of systems.')


        # Model for ambient temperature
        self.ambtemp_model = ambtemp_model


        # Configure action space or throw error if unknown
        if act_space in ['D10', 'D20', 'D50', 'D100', 'D200', 'D500']:
            sizeActionSpace = int(act_space[1:])  # conv to useable number
            self.action_space = spaces.Discrete(float(sizeActionSpace))
        elif act_space is 'C':
            self.action_space = spaces.Box(np.array([0.]),  # todo: set range
                                           np.array([100.]),
                                           dtype=np.float64)
        else:
            raise ValueError('Error: unknown act_space specifier.')


        # Reward function
        self.reward_type = reward_type


        # Configure time-step size or throw error if unknown
        if timestep_size[0] is 't':
            self.timestep = int(timestep_size[1:])  # Defines the number of seconds in one time-step

        else:
            raise ValueError(
                'Error: timestep_size must start with t, specifier bad')

        self.t_int_step = 0.1  # seconds between integration steps
        self.t = np.arange(0, self.timestep, self.t_int_step) # time array for odeint


        # Configure observation space : T_can: [15,90]C; T_amb: [0,50]C
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

    # Configure ambient temperature model or raise error if model is unknown
    def T_amb(self, time):
        """Returns ambient temperature based on ambient temperature model defined in the specific environment """
        if self.ambtemp_model is 'Tcon':
            return TambModels.TConstant()
        elif self.ambtemp_model is 'Tsin':
            return TambModels.TSine(self.elapsed_steps, time, self.timestep)
        elif self.ambtemp_model is 'Trand':
            return TambModels.TRandom()
        elif self.ambtemp_model is 'Tsinrand':
            return TambModels.TSineRandom(self.elapsed_steps, time, self.timestep)
        else:
            raise ValueError('Error: unknown ambient temperature model')


    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.elapsed_steps += 1
        T_can = self.state[0]
        self.P_heat = action

        #  gets final value after integration
        T_can_updated = float(odeint(
            self.vac_can, T_can, self.t)[int(self.timestep/self.t_int_step) - 1])

        self.state = np.array([T_can_updated,
                               self.T_amb(self.elapsed_steps*self.timestep)])

        done =  not self.observation_space.contains(self.state)
        done = bool(done)   # kill run if railed

        # todo: hard codeing reward, need to refactor rewards as class
        T_setpoint = 45
        if not done:
            if self.reward_type is 'Rw10':
                reward = rewardType.RewardWindow10(T_can_updated,T_setpoint)
            elif self.reward_type is 'Rw4':
                reward = rewardType.RewardWindow4(T_can_updated, T_setpoint)
            elif self.reward_type is 'Rexp':
                reward = rewardType.RewardExp(T_can_updated, T_setpoint)
            elif self.reward_type is 'Rquad':
                reward = rewardType.RewardQuadratic(T_can_updated, T_setpoint)
            elif self.reward_type is 'Rrecquad':
                reward = rewardType.RewardReciprocalQuadratic(T_can_updated, T_setpoint)
            else:
                raise ValueError('Error: unknown reward function')

        return self.state, reward, done, {}
