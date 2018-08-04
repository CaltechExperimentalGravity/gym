# Test Environments for Thermal control

This section contains the python code to create simulations of the physical systems for thermal control as gym environments on which RL algorithms can be trained and tested.

## Default Envs

Environments with default parameters can be accessed through `VacCan-v0`, `VacCanC-v0` for the discrete and continuous versions respectively.

```
pip install -e .['VacCan-v0']
```

For environment with continuous actuation, install using:
```
pip install -e .['VacCanC-v0']
```

* `VacCan-v0`, `VacCanC-v0` env : simulation of thermal dynamics of the vacuum can considering the conduction through the foam and heating.

  *Reward*: 0.1 for every time step spent in interval (43,47) C.

  *Ambient temperature*: Oscillates around 20 C with an amplitude of 5 C, depending on number of steps elapsed.

## Parametrised Envs

These are different envs for thermal systems with different parameters for training RL algorithms for accurate temperature control.

Each parametrised environment initially runs from `Run.py` and refers to the models in `Models.py` based on the parameters described in its name.

The attributes of the environment that can be varied are:
1. `<thermal_param>` The set of thermal parameters for testing: `Vaccan` or `Seism` referring to the vacuum can thermal system or the seismometer thermal system.

2. `<act_space>` Size and nature of the action space, which can be discrete with various sizes dividing up the interval between 0 and 100 W: `D10`, `D20`, `D50`, `D100`, `D200`, `D500`
or continuous between 0 and 100 W: `C` .

3. `<reward_type>` Type of reward function, which include windows of size 10 C and 4 C: `Rw10`, `Rw4`;
quadratic: `Rquad`;
exponential: `Rexp`

4. `<ambtemp_models>` Model for ambient temperature, which can be constant, random, sinusoidal, or noisy sinusoidal: `Tcon`, `Trand`, `Tsin`, `Tsinrand`

5. `<timestep_size>` Size of each time step in the game in seconds: `t1`, `t10`, `t30`, `t60`, `t100`


  In order to install a particular one of these envs, the following scheme is used:

`pip install -e .['<thermal_param>_<act_space>_<reward_type>_<ambtemp_models>_<timestep_size>-v0']`

As an example:

`pip install -e .['Vaccan_D200_Rquad_Tsin_t30-v0']`

Then, using python, the environment can be made and the state reset as follows:
   `env = gym.make('Vaccan_D200_Rquad_Tsin_t30-v0')`

   `env.reset()`
