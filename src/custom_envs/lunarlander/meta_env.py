import math
import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from src.custom_algorithms.cleanppo.cleanppo import CLEANPPO


class MetaEnv(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self,
                 rl_model_best: str,
                 render_mode=render_mode):
        self.ROOT_DIR = "."

        ### ACTION SPACE ###
        # one action to decide which task to control
        # action 0 --> agent 1 can be controlled
        # action 1 --> agent 2 can be controlled
        self.action_space = gym.spaces.Discrete(2)

        ### OBSERVATION SPACE ###
        # 8 dimensions for each subagent:
        #   the coordinates of the lander in x & y --> low:-2,5 high:2,5
        #   linear velocities in x & y --> low:-10 high:10
        #   angle --> low:-6.2831855 high:6.2831855
        #   angular velocity --> low:-10 high:10
        #   two booleans that represent whether each leg is in contact with the ground or not --> low:0 high:1
        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        self.observation_space = gym.spaces.Dict(
            {"agent_one": gym.spaces.Box(low=low, high=high),
             "agent_two": gym.spaces.Box(low=low, high=high),
             # rewards are unbounded but target is min 200 for each subagent
             "reward_one": gym.spaces.Box(low=-np.inf,
                                          high=np.inf,
                                          shape=(1,),
                                          dtype=np.float32),
             "reward_two": gym.spaces.Box(low=-np.inf,
                                          high=np.inf,
                                          shape=(1,),
                                          dtype=np.float32),
             # state of subagents
             #   0 --> the agent is still playing
             #   1 --> the agent has won the game
             #   2 --> the agent has lost the game
             "state_one": gym.spaces.Box(low=0,
                                         high=2,
                                         shape=(1,),
                                         dtype=np.int64),
             "state_two": gym.spaces.Box(low=0,
                                         high=2,
                                         shape=(1,),
                                         dtype=np.int64),
             # sum of subagent rewards
             "meta_reward": gym.spaces.Box(low=-np.inf,
                                          high=np.inf,
                                          shape=(1,),
                                          dtype=np.float32),
             # selected subagent (0 or 1)
             "meta_action": gym.spaces.Box(low=0,
                                           high=1,
                                           shape=(1,),
                                           dtype=np.int64),
             }
        )

        # logger
        tmp_path = "/tmp/sb3_log/"
        self.logger = configure(tmp_path, ["stdout", "csv"])
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # load the trained subagents
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             f"../../../policies/{rl_model_best}"), "rb"
        ) as file:
            print("start loading agents", file)
            self.trained_agent_one = CLEANPPO.load(path=file,
                                                           env=make_vec_env("CustomLunarLander-v2",
                                                                            n_envs=1,
                                                                            env_kwargs={
                                                                                "gravity":-5.0,
                                                                                "enable_wind":False,
                                                                                "wind_power":10.0,
                                                                                "render_mode": None}))
            self.trained_agent_one.set_logger(logger=self.logger)
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             f"../../../policies/{rl_model_best}"), "rb"
        ) as file:
            self.trained_agent_two = CLEANPPO.load(path=file,
                                                           env=make_vec_env("CustomLunarLander-v2",
                                                                            n_envs=1,
                                                                            env_kwargs={
                                                                                "gravity":-5.0,
                                                                                "enable_wind":False,
                                                                                "wind_power":10.0,
                                                                                "render_mode": None}))
            self.trained_agent_two.set_logger(logger=self.logger)
            print("finish loading agents")

        # state of subagents
        self.last_state_of_agent_one = self.trained_agent_one.env.reset()
        self.last_state_of_agent_two = self.trained_agent_two.env.reset()
        self.total_reward_of_agent_one = 0.0
        self.total_reward_of_agent_two = 0.0

        # state of meta-agent
        self.state = {
            "agent_one": self.last_state_of_agent_one,
            "agent_two": self.last_state_of_agent_two,
            "reward_one" : 0.0,
            "reward_two" : 0.0,
            "state_one": 0,
            "state_two": 0,
            "meta_reward": 0.0,
            "meta_action": 0
        }

        # for rendering
        # FIXME: display sind nicht getrennt
        # if not self.render_mode is None:

        # counter
        self.episode_counter = 0
        self.step_counter = 0
        self.counter_without_switch = 0
        self.last_action = 0

    def step(self, action: int):
        """
           action: selects the subagent
                    0: agent one
                    1: agent two
        """
        # automatically switch to the other agent if the selected one has already won the game
        # start abruptly a new episode only if one subagent loses the game while the other is still playing
        if action == 0 and self.state["state_one"] == 1:
            action = 1
            # print("PLAY AGENT TWO!")
        if action == 1 and self.state["state_two"] == 1:
            action = 0
            # print("PLAY AGENT ONE!")

        # task switch
        if self.last_action == action:
            self.counter_without_switch += 1
        else:
            self.counter_without_switch = 0
            self.last_action = action

        # action 0: agent one selected
        # action 1: agent two selected
        match action:
            case 0:
                # agent one
                active_agent = self.trained_agent_one
                active_last_state = self.last_state_of_agent_one
                inactive_agent = self.trained_agent_two
                inactive_last_state = self.last_state_of_agent_two
                self.current_task = 0
                self.state["meta_action"] = 0
            case 1:
                # agent two
                active_agent = self.trained_agent_two
                active_last_state = self.last_state_of_agent_two
                inactive_agent = self.trained_agent_one
                inactive_last_state = self.last_state_of_agent_one
                self.current_task = 1
                self.state["meta_action"] = 1

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1")

        ### ACTIVE AGENT ###
        # predict next action
        # action_of_active_agent[0] is the next action
        action_of_active_agent, _ = active_agent.predict(active_last_state, deterministic=True)

        # test for step outputs
        #result = active_agent.env.step(np.array(action_of_active_agent))
        #print("Result from step:", result, "Type:", type(result), "Length:", len(result))

        # perform the action
        # only four return value because DummyVecEnv only returns observation, reward, done, info
        # because of that terminated (won) and truncated (lost) info is in active_info
        (active_obs,
         active_reward,
         _,
         active_info) = active_agent.env.step(np.array(action_of_active_agent))

        # extract terminated and truncated values from active_info
        active_terminated = active_info[0]["terminated"]
        active_truncated = active_info[0]["truncated"]

        # print("new_obs:", new_obs,
        #       "active_reward:", active_reward,
        #       "active_terminated:", active_terminated,
        #       "active_truncated:", active_truncated)

        ### INACTIVE AGENT ###
        # perform action 0, which means "do nothing"
        # meta agent does not see actual state and reward
        # this step runs only if inactive agent is still playing.
        if not any([inactive_agent == self.trained_agent_one and self.state["state_one"] == 1,
                    inactive_agent == self.trained_agent_two and self.state["state_two"] == 1]):
            # only four return value because DummyVecEnv only returns observation, reward, done, info
            # because of that terminated (won) and truncated (lost) info is in inactive_info
            (inactive_obs,
            inactive_reward,
            _,
            inactive_info) = inactive_agent.env.step(np.array([0]))

            # print("inactive_obs:", inactive_obs)

            # extract terminated and truncated values from inactive_info
            inactive_terminated = inactive_info[0]["terminated"]
            inactive_truncated = inactive_info[0]["truncated"]
        # if the inactive agent has won the game, it no longer takes steps (its game is over)
        else:
            inactive_obs = inactive_last_state
            inactive_terminated = True
            inactive_truncated = False
            inactive_reward = 0

        ### UPDATE ###
        # update the obs of active agent and reward of both agents
        match action:
            # agent one
            case 0:
                # update states
                # updated state of inactive agent is not visible to the meta-agent
                self.last_state_of_agent_one = active_obs
                self.state["agent_one"] = active_obs
                self.last_state_of_agent_two = inactive_obs
                if active_terminated:
                    self.state["state_one"] = 1
                    # print("AGENT ONE WON")
                elif active_truncated:
                    self.state["state_one"] = 2
                # meta agent only knows if inactive agent is still playing
                if inactive_terminated:
                    self.state["state_two"] = 1
                    # print("AGENT TWO WON")
                elif inactive_truncated:
                    self.state["state_two"] = 2
                # update rewards
                # updated reward of inactive agent is not visible to the meta-agent
                self.total_reward_of_agent_one += active_reward
                self.state["reward_one"] = self.total_reward_of_agent_one
                self.total_reward_of_agent_two += inactive_reward
            # agent two
            case 1:
                # update states
                # updated state of inactive agent is not visible to the meta-agent
                self.last_state_of_agent_two = active_obs
                self.state["agent_two"] = active_obs
                self.last_state_of_agent_one = inactive_obs
                if active_terminated:
                    self.state["state_two"] = 1
                    # print("AGENT TWO WON")
                elif active_truncated:
                    self.state["state_two"] = 2
                # meta agent only knows if inactive agent is still playing
                if inactive_terminated:
                    self.state["state_one"] = 1
                    # print("AGENT ONE WON")
                elif inactive_truncated:
                    self.state["state_one"] = 2
                # update rewards
                # updated reward of inactive agent is not visible to the meta-agent
                self.total_reward_of_agent_two += active_reward
                self.state["reward_two"] = self.total_reward_of_agent_two
                self.total_reward_of_agent_one += inactive_reward
            case _:
                raise ValueError("action must be 0, 1")

        self.step_counter += 1
        self.state["meta_reward"] = self.total_reward_of_agent_one + self.total_reward_of_agent_two
        reward = active_reward + inactive_reward

        win_counter = -1
        #meta_reward = 0
        if any([active_terminated and inactive_terminated,
                active_truncated or inactive_truncated]):
            win_counter = 0
            #meta_reward = self.state["meta_reward"]
            if self.state["state_one"] == 1:
                win_counter += 1
            if self.state["state_two"] == 1:
                win_counter += 1

        # all infos in this function
        info = {"active_agent": action,
                "active_reward": active_reward,
                "active_terminated": active_terminated,
                "active_truncated": active_truncated,
                "inactive_reward": inactive_reward,
                "inactive_terminated": inactive_terminated,
                "inactive_truncated": inactive_truncated,
                "total_reward_of_agent_one": self.total_reward_of_agent_one,
                "total_reward_of_agent_two": self.total_reward_of_agent_two,
                "meta_reward": self.state["meta_reward"],
                "episode_counter": self.episode_counter,
                "step_counter": self.step_counter,
                "last_action": self.last_action,
                "counter_without_switch": self.counter_without_switch,
                "win_counter": win_counter}

        # ### TEST OF RESULT ###
        # # active agent number
        # # states of both subagents
        # # terminated infos of both subagents
        # # truncated infos of both agents
        # # reward of all agents (subagents and meta agent)
        print("step_counter:", self.step_counter)
        print("active agent:", self.state["meta_action"])
        print("state_one", self.last_state_of_agent_one,
              "state_two", self.last_state_of_agent_two),
        print("state one for meta", self.state["state_one"],
              "state two for meta", self.state["state_two"])
        print("active_terminated:", active_terminated,
              "inactive_terminated:", inactive_terminated)
        print("active_truncated;", active_truncated,
              "inactive_truncated:", inactive_truncated)
        print("total reward_one:", self.total_reward_of_agent_one,
              "total reward_two:", self.total_reward_of_agent_two,
              "meta_reward:", self.state["meta_reward"])
        print("active_reward:", active_reward,
              "inactive_reward:", inactive_reward,
              "reward:", reward)

        #if not self.render_mode is None:

        return (
            self.state,
            reward,
            active_terminated and inactive_terminated,
            active_truncated or inactive_truncated,
            info
        )

    def reset(self, seed=None, options=None):
        ### TEST OF EPISODE-RESULT ###
        # states of both subagents
        # reward of all agents (subagents and meta agent)
        # print("episode_counter:", self.episode_counter,
        #       "state_one", self.state["state_one"],
        #       "state_two", self.state["state_two"],
        #       "reward_one:", self.total_reward_of_agent_one,
        #       "reward_two:", self.total_reward_of_agent_two,
        #       "meta_reward:", self.state["meta_reward"])

        # reset state of subagents
        # only one return value because DummyVecEnv only returns one observation
        self.last_state_of_agent_one = self.trained_agent_one.env.reset()
        self.last_state_of_agent_two = self.trained_agent_two.env.reset()
        self.total_reward_of_agent_one = 0.0
        self.total_reward_of_agent_two = 0.0

        # reset state of meta-agent
        self.state = {
            "agent_one": self.last_state_of_agent_one,
            "agent_two": self.last_state_of_agent_two,
            "reward_one": 0.0,
            "reward_two": 0.0,
            "state_one": 0,
            "state_two": 0,
            "meta_reward": 0.0,
            "meta_action": 0
        }

        # reset counter
        self.episode_counter += 1
        self.step_counter = 0
        self.counter_without_switch = 0
        self.last_action = 0

        return self.state, {}
