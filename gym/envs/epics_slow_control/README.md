# Test Environments for Thermal control

Contains the python code to interface a real physical system as a gym environment on which RL algorithms can be trained and tested

## Available Envs
* `FSSSlowCtrl` env :  access to laser slow controls to sweep laser frequency and engauge lock when close to resonance of cavity. 

This environment can be installed using:

```
pip install -e .['FSSSlowCtrl-v0']
```

NOTE: this enviroment must be implemented on a machine with pyepics installed and inside lab network. It might be possible to access with VPN or port forwarding into the lab.  The training scheme should be designed in such a way that it is paused when epics channel C3:PSL-SCIENCEMODE_EN is engaged. This mean the experiment will go into training mode when not in use.
