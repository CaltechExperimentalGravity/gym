# Test Environments for Thermal control

Contains the python code to create simulations of the physical system as a gym environment on which RL algorithms can be trained and tested

## Available Envs

Each environment with discrete actuation can be installed using:

```
pip install -e .['VacCan-vx']
```
where x is the version number.

For environments with continuous actuation, install using:
```
pip install -e .['VacCanC-vx']
```
for version x.



* `VacCan-v0`, `VacCanC-v0` env : simulation of thermal dynamics of the vacuum can considering the conduction through the foam and heating.

  *Reward*: 0.1 for every time step spent in interval (40,50) C.
  
  *Ambient temperature*: Oscillates around 20 C with an amplitude of 5 C, depending on number of steps elapsed.

  
* `VacCan-v1`:

  *Reward*: 0.1 for every time step spent in interval (43,47) C.
  
  *Ambient temperature*: Oscillates around 20 C with an amplitude of 5 C, depending on number of steps elapsed.
  
* `VacCanC-v1`:

  *Reward*: 0.1 for every time step spent in interval (43,47) C.

  *Ambient temperature*: Random value between 19 and 21 C.

* `VacCan-v2`, `VacCanC-v2`:

  *Reward*: 0.1 for every time step spent in interval (42,45] C.

  *Ambient temperature*: Oscillates around 20 C with an amplitude of 5 C, depending on number of steps elapsed.

* `VacCan-v3`, `VacCanC-v3`:

  *Reward*: $ 1 - \frac{(T-T_{setpoint})^2}{T_{setpoint}}^2 $

  *Ambient temperature*: Oscillates around 20 C with an amplitude of 5 C, depending on number of steps elapsed.

* `VacCan-v4`, `VacCanC-v4`:

  *Reward*: $ e^{-\frac{{T-T_{setpoint}}^2}{2*T_{setpoint}}} $
