import gym
from gym import logger
import gym.spaces as spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint

##
import matplotlib as mpl
import matplotlib.cm as cm

from os import path


class VacCanTestEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 2
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
        self.action_space_dim = 20
        self.action_space = spaces.Discrete(self.action_space_dim)
        self.observation_space = spaces.Box(np.array([15.0, 0.0]),
                                            np.array([60.0, 50.0]),
                                            dtype=np.float64)

        self.seed()
        # self.state = None
        self.steps_beyond_done = None
        self.elapsed_steps = 0
        self.reset()

        self.viewer = None


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
        return 5*np.sin(2*np.pi*(self.elapsed_steps*10. + time)/(6*3600)) + 20.


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

        done = T_can_updated < 15. or T_can_updated > 60. or self.elapsed_steps == 100
        done = bool(done)

        if not done:
            if self.state[0] > 43. and self.state[0] < 47.:
                reward = 1
            else:
                reward = 0
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
        self.state = [self.np_random.uniform(low=15, high=60), self.T_amb(0)]
        self.steps_beyond_done = None
        return np.array(self.state)


    def render(self, mode='human'):

        norm = mpl.colors.Normalize(15,60)
        m = cm.ScalarMappable(norm=norm, cmap=cm.hot)


        screen_width = 600
        screen_height = 400



        can_rad = 80.0
        foam_rad = 120.0
        polewidth = 10.0
        polelen = 100.0

        if self.viewer is None:
            from gym.envs.temp_ctrl import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            axleoffset = 200

            can = rendering.make_circle(can_rad)
            #can.set_color(.1,.4,.6)

            self.can_tran = rendering.Transform(translation = (screen_width/3.5,screen_height/2))
            can.add_attr(self.can_tran)


            foam = rendering.make_circle(foam_rad)
            foam.set_color(.6,.4,.6)

            foam.add_attr(self.can_tran)
            self.viewer.add_geom(can)
            self.viewer.add_geom(foam)
            self.viewer.add_geom(can)



            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(screen_width*2.5/3., axleoffset))
            pole.add_attr(self.poletrans)

            self.viewer.add_geom(pole)


            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)

            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)

            fname = path.join(path.dirname(__file__), "assets/scale.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.img.add_attr(self.poletrans)
            self.viewer.add_geom(self.img)

        if self.state is None: return None

        x = self.state
        h = self.P_heat
        can_color = np.asarray(m.to_rgba(x[0]))[:3]
        foam_color = np.asarray(m.to_rgba(x[1]))[:3]

        can.set_color(can_color[0], can_color[1], can_color[2])
        foam.set_color(foam_color[0], foam_color[1], foam_color[2])

        self.viewer.add_geom(can)
        self.viewer.add_geom(foam)
        self.viewer.add_geom(can)

        self.poletrans.set_rotation(h*np.pi/self.action_space_dim -np.pi/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
