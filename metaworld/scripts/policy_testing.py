import random
import time

import numpy as np

import metaworld
from metaworld.policies.sawyer_door_lock_v2_policy import (
    SawyerDoorLockV2Policy as policy,
)

np.set_printoptions(suppress=True)

seed = 42
env_name = "door-lock-v2"

random.seed(seed)
ml1 = metaworld.MT50(seed=seed)
env = ml1.train_classes[env_name]()
task = [t for t in ml1.train_tasks if t.env_name == env_name][0]
env.set_task(task)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
obs = env.reset()

p = policy()
count = 0
done = False

states = []
actions = []
next_states = []
rewards = []

dones = []
info = {}

while count < 500 and not done:
    action = p.get_action(obs)
    next_obs, _, _, _, info = env.step(action)
    # env.render()
    print(count, next_obs)
    if int(info["success"]) == 1:
        done = True
    obs = next_obs
    time.sleep(0.02)
    count += 1

print(info)
