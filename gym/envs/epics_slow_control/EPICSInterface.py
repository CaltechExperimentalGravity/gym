import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
from time import sleep

# Channel access
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
        'render.modes': ['human']
    }

    def __init__(self):
        self.t_step = 0.1  # seconds between state updates

        self.monitor1 = "C3:PSL-SCAV_REFL_DC"
        self.monitor2 = "C3:PSL-SCAV_TRANS_DC"
        self.monitor3 = "C3:PSL-SCAV_FSS_FASTMON"
        self.loopStateEnable = "C3:PSL-SCAV_FSS_RAMP_EN"
        self.actuator = "C3:PSL-SCAV_FSS_SLOWOUT"

        # Set-point of process
        self.setpoint = 1.0  # volts

        # Set bounds on search range hit the edge and you fail episode
        self.ActuatorBounds = np.array[3.1, 3.8]

        # Observation chan one win case threshold
        self.TransLockThresh = 2.0
        self.RelfLockThresh = 1.7
        self.FSSFastBounds = np.array[0.8, 1.2]

        # Establish action space
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(np.array([0.0, 5.0]),
                                            np.array([-17.0, 17.0]),
                                            dtype=np.float32)  # EPICS lim 4sf
# TODO: seed might not be needed
        self.seed()
        self.state = None
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

        monitor1_value = caget(self.monitor1)
        monitor2_value = caget(self.monitor2)
        monitor3_value = caget(self.monitor3)
        actuator_value = caget(self.actuator)

        # action tree
        ''' Here action is split into 2x3 options. Engage switch is either on
            off, and the slow laser control voltage is either nudged
            up/down/left alone.  Its not clear yet if this will be easy to seed
            a starting point that will be able to find a good strategy.
            '''
        # # TODO: add rate limit in case flicking PZT quickly might damage it
        if action < 3:
            actuator_value = actuator_value + 0.0001 * (action - 1)
            caput(self.actuator, actuator_value)
            caput(self.loopStateEnable, 0)
        elif action >= 3:
            actuator_value = actuator_value + 0.0001 * (action - 4)
            caput(self.actuator, actuator_value)
            caput(self.loopStateEnable, 1)

        self.state = (monitor1_value, monitor1_value, monitor1_value)
        sleep(self.t_step)  # rate limit channel access
        # TODO: This is a dumb end condition if bound can be set by obser space
        done = actuator_value < self.ActuatorBounds[0] or \
            actuator_value > self.ActuatorBounds[1]
        done = bool(done)

        # reward schedule
        if not done:
            if self.state[0] < 1.5:
                width = 0.5
                # mountain/triangle shapped reward function
                reward = 1.0 + np.clip(
                    1 - 2 / width * np.abs(self.state[2] - self.setpoint), 0, 1)
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
        # self.state = [self.np_random.uniform(low=15, high=30), self.T_amb(0)]
        resetChannVal = self.np_random.uniform(low=3.1, high=3.8, size=(1,))
        caput(self.loopStateEnable, 0)
        caput(self.actuator, resetChannVal)
        #  self.state = 1.0  # dummy state to start
        self.steps_beyond_done = None
        return np.array(self.state)
