# rand_param_envs (using MuJoCo 2.0)

Fork from https://github.com/dennisl88/rand_param_envs, which provides random parameter environments using gym 0.7.4 and mujoco-py 0.5.7 and is used in the [ProMP implementation](https://github.com/jonasrothfuss/ProMP) by Rothfuss et al. (2018). Unfortunately, the above doesn't work on newer Macs because it uses MuJoCo 1.31 (due to a well-known incompatibility with older py-mujoco versions, see, e.g., [here](https://github.com/openai/mujoco-py/issues/36) or [here](http://www.mujoco.org/forum/index.php?threads/trying-to-run-the-mjpro131-can-not-open-disk.3439/)).

This fork works with gym 0.15.4 and mujoco-py 2.0.2.9 (using MuJoCo 2.0).

Tested running 
``` 
python run_scripts/ppo_single_run.py
```
from the [ProMP repo](https://github.com/jonasrothfuss/ProMP/tree/full_code) (`full_code` branch).


