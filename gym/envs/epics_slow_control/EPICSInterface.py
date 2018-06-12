import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
# from scipy.integrate import odeint
from time import sleep

# Channel access
# from ezca import Ezca
from epics import caget, caput

class EPICSInterfaceEnv(gym.Env):
    ''' This environment interfaces real world slow EPICS channels into the
        standard form of a gym environment. Real training can then take place
        using real world systems.

        This environment is still beta and care should be taken that any
        possible actions of the environement in the real world are safe.

        In this initial implementation we are interfacing with laser slow
        controls of the Coating Thermal Noise experiment.  Channels are
        hard coded for now.  The goal is to achieve fastest possible lock
        of pair of cavities.  Avaliable actions are binary lock switch and
        laser slow control voltage.  Observations will be the reflected and
        transmitted power of cavity.  Eventually we may also include the task
        of holding the slow laser frequency close to a set point.

        Its note clear if we can use continuous variables for the action state
        atm.  For now we will implment steps as actions to increment epics
        up and down.  As a first pass this will be up/down/stay put.  But
        the next step would be to implement a range of values for size fo step.
        '''

    metadata = {
        'render.modes':['human']
    }

    def __init__(self):
        # RCPID = Ezca(ifo=None, logger=False)  # ezca python access to EPICS
        self.d = 5.08e-2
        self.t_step = 0.1  # seconds between state updates
        self.t_max = 10  # 10 seconds = 1 time-step

        # Set-point of process
        self.T_setpoint = 1.0  # volts

        # Set bounds on search range hit the edge and you fail episode
        self.ActuatorUB = 3.8
        self.ActuatorLB = 3.1

        # Observation chan one win case threshold
        self.TransLockThresh = 2.0
        self.RelfLockThresh = 0.8
        self.FSSFastUB = 1.2
        self.FSSFastLB = 0.8

        # Establish action space
        self.action_space = spaces.Discrete(6)
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


# Simulates reaction
    def step(self, action):
        assert self.action_space.contains(action), \
                "%r (%s) invalid" % (action, type(action))

        self.elapsed_steps += 1
        ProcessChan = "C3:PSL-SCAV_TRANS_DC"
        # ProcessVal = RCPID.read(ProcessChan)
        ProcessVal = caget(ProcessChan)

        self.P_heat = action

        #  gets final value after integration

        self.state = ProcessVal
        sleep(2)
        done = ProcessVal > 1.5
        done = bool(done)

        if not done:
            if self.state[0] > 1.5 or self.state[0] < 4.0:
                reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
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
        # self.state = [self.np_random.uniform(low=15, high=30), self.T_amb(0)]
        self.state = 1.0  # dummy state to start
        self.steps_beyond_done = None
        return np.array(self.state)
