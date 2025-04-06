'''
    THIS META-AGENT-EVALUATION IS IMPLEMENTED FOR THE LUNAR LANDER SETUP
    Adapted from Annika Österdiekhoff's Moon Lander setup
'''

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from src.custom_envs.register_envs import register_custom_envs

### LOAD AGENT ###
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Verwendetes Gerät:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# register meta_env
register_custom_envs()

vec_env = make_vec_env("MetaLunarLander-pretrained-v0", n_envs=1)

# define policy and parameter
policy = "MultiInputPolicy"
kwargs = {"verbose": True }

# initialize PPO algorithm
algorithm = PPO(policy, env=vec_env, **kwargs)

# load pretrained meta agent
model = algorithm.load(
    path="../policies/meta_finished_sig_0.zip",
    env=vec_env,
    **kwargs
)

# logger
tmp_path = "/tmp/sb3_log/"
logger = configure(tmp_path, ["stdout", "csv"])
model.set_logger(logger=logger)

# link env to the loaded meta agent
vec_env = model.get_env()

### EVALUATION ###
num_episodes = 1000

# save outputs as file
file_name = "meta_base_1000"
with open(file_name, "a") as file:

    for i in range(num_episodes):
        # reset env at the beginning of each episode
        observation = vec_env.reset()

        terminated = False
        while not terminated:
            # predict next action
            action, _state, = model.predict(observation, deterministic=True)
            # take a step
            observation, reward, terminated, info = vec_env.step(action)

            # extract all reward values and win value
            if info[0]["reward_one"] == 0.0:
                reward_one = 0.0
            else:
                reward_one = np.around(info[0]["reward_one"].item(), 3)

            if info[0]["reward_two"] == 0.0:
                reward_two = 0.0
            else:
                reward_two = np.around(info[0]["reward_two"].item(), 3)

            if info[0]["meta_reward"] == 0.0:
                meta_reward = 0.0
            else:
                meta_reward = np.around(info[0]["meta_reward"].item(), 3)

        # print and save
        print(np.around(info[0]["step_counter"]),
              reward_one,
              reward_two,
              meta_reward,
              info[0]["win_counter"], file=file)

# close env after evaluation
vec_env.close()