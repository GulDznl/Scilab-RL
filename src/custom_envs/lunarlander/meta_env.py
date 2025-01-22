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
                 rl_model_best_with_pause: str,
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
                             f"../../../policies/{rl_model_best_with_pause}"), "rb"
        ) as file:
            print("start loading agents", file)
            self.trained_agent_one = CLEANPPO.load(path=file,
                                                           env=make_vec_env("CustomLunarLander-v2",
                                                                            n_envs=1,
                                                                            env_kwargs={
                                                                                "gravity": -5.0,
                                                                                "enable_wind": False,
                                                                                "wind_power": 10.0,
                                                                                "render_mode": None}))
            self.trained_agent_one.set_logger(logger=self.logger)
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             f"../../../policies/{rl_model_best_with_pause}"), "rb"
        ) as file:
            self.trained_agent_two = CLEANPPO.load(path=file,
                                                           env=make_vec_env("CustomLunarLander-v2",
                                                                            n_envs=1,
                                                                            env_kwargs={
                                                                                "gravity": -5.0,
                                                                                "enable_wind": False,
                                                                                "wind_power": 10.0,
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
        # subagent selection sequence in an episode
        self.switch_sequence = []

        # parameters for choice-based reward
        self.last_reward_of_agent_one = 0.0
        self.last_reward_of_agent_two = 0.0

        # for rendering
        # FIXME: display sind nicht getrennt
        # if not self.render_mode is None:

        # counter
        self.episode_counter = 0
        self.step_counter = 0
        self.counter_without_switch = 0
        #self.last_counter_without_switch = 0
        self.last_action = 0

    def step(self, action: int):
        """
           action: selects the subagent
                    0: agent one
                    1: agent two
        """
        # automatically switch to the other agent if the selected one has already won the game
        # or counter_without_switch more than 7
        # start abruptly a new episode only if one subagent loses the game while the other is still playing
        feedback = 0
        match action:
            case 0:
                # agent one
                if (self.state["state_one"] == 1
                        or self.counter_without_switch > 7):
                    action = 1
                    #self.last_counter_without_switch = 1
                    feedback = -50
                    print("PLAY AGENT TWO!")
            case 1:
                # agent two
                if (self.state["state_two"] == 1
                        or self.counter_without_switch > 7):
                    action = 0
                    #self.last_counter_without_switch = 1
                    feedback = -50
                    print("PLAY AGENT ONE!")

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1")

        # save the action
        self.switch_sequence.append(action)

        # task switch
        if self.last_action == action:
            self.counter_without_switch += 1
        else:
            #self.last_counter_without_switch = self.counter_without_switch + 0
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
                inactive_last_reward = self.total_reward_of_agent_two
                self.state["meta_action"] = 0
            case 1:
                # agent two
                active_agent = self.trained_agent_two
                active_last_state = self.last_state_of_agent_two
                inactive_agent = self.trained_agent_one
                inactive_last_state = self.last_state_of_agent_one
                inactive_last_reward = self.total_reward_of_agent_one
                self.state["meta_action"] = 1

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1")

        ### ACTIVE AGENT ###
        # predict next action
        # action_of_active_agent[0] is the next action
        action_of_active_agent, _ = active_agent.predict(active_last_state, deterministic=True)

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

            # extract terminated and truncated values from inactive_info
            inactive_terminated = inactive_info[0]["terminated"]
            inactive_truncated = inactive_info[0]["truncated"]
        # if the inactive agent has won the game, it no longer takes steps (its game is over)
        else:
            inactive_obs = inactive_last_state
            inactive_terminated = True
            inactive_truncated = False
            inactive_reward = inactive_last_reward

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
        #self.state["meta_reward"] = active_reward

        ### CHOICE-BASED REWARD ###
        # meta agent gets the reward differences of subagents as its reward
        reward_diff_of_agent_one = abs(self.last_reward_of_agent_one - self.total_reward_of_agent_one)
        reward_diff_of_agent_two = abs(self.total_reward_of_agent_two - self.last_reward_of_agent_two)
        reward = reward_diff_of_agent_one + reward_diff_of_agent_two
        print("last_reward_of_agent_one:", self.last_reward_of_agent_one)
        print("last_reward_of_agent_two:", self.last_reward_of_agent_two)
        print("reward:", reward)
        # update last rewards of subagents
        # without +0, last rewards automatically update with total
        # rewards during the subagents' update process above (unknown why)
        self.last_reward_of_agent_one = self.total_reward_of_agent_one + 0
        self.last_reward_of_agent_two = self.total_reward_of_agent_two + 0
        print("last_reward_of_agent_one:", self.last_reward_of_agent_one)
        print("last_reward_of_agent_two:", self.last_reward_of_agent_two)

        # get number of successful agents at episode end
        win_counter = -1
        if any([active_terminated and inactive_terminated,
                active_truncated or inactive_truncated]):
            # print("self.step_counter:", self.step_counter)
            # print("meta_reward:", self.state["meta_reward"])
            # print("total_reward_of_agent_one:", self.total_reward_of_agent_one)
            # print("total_reward_of_agent_two:", self.total_reward_of_agent_two)
            # print("self.switch_sequence:", self.switch_sequence)
            # print("state_of_agent_one:", self.state["state_one"])
            # print("state_of_agent_two:", self.state["state_two"])
            win_counter = 0
            if self.state["state_one"] == 1:
                win_counter += 1
            if self.state["state_two"] == 1:
                win_counter += 1

        # all infos in this function
        info = {"episode_counter": self.episode_counter,
                "step_counter": self.step_counter,
                "win_counter": win_counter,
                "counter_without_switch": self.counter_without_switch,
                "last_action": self.last_action,
                "active_agent": action,
                "active_reward": active_reward,
                "active_terminated": active_terminated,
                "active_truncated": active_truncated,
                "inactive_reward": inactive_reward,
                "inactive_terminated": inactive_terminated,
                "inactive_truncated": inactive_truncated,
                "total_reward_of_agent_one": self.total_reward_of_agent_one,
                "total_reward_of_agent_two": self.total_reward_of_agent_two,
                "state_of_agent_one": self.state["agent_one"],
                "state_of_agent_two": self.state["agent_two"],
                "reward_one": self.state["reward_one"],
                "reward_two": self.state["reward_two"],
                "state_one": self.state["state_one"],
                "state_two": self.state["state_two"],
                "meta_reward": self.state["meta_reward"],
                "meta_action": self.state["meta_action"]}

        ### TEST OF RESULT ###
        print("STEP ENDE")
        print("episode_counter:", self.episode_counter)
        print("step_counter:", self.step_counter)
        print("win_counter:", win_counter)
        print("counter_without_switch:", self.counter_without_switch)
        print("last_action:", self.last_action)
        print("active_agent:", action)
        print("active_reward:", active_reward)
        print("active_terminated:", active_terminated)
        print("active_truncated:", active_truncated)
        print("inactive_reward:", inactive_reward)
        print("inactive_terminated:", inactive_terminated)
        print("inactive_truncated:", inactive_truncated)
        print("total_reward_of_agent_one:", self.total_reward_of_agent_one)
        print("total_reward_of_agent_two:", self.total_reward_of_agent_two)
        print("Eigentlich state_of_agent_one:", self.last_state_of_agent_one)
        print("Eigentlich state_of_agent_two:", self.last_state_of_agent_two)
        print("last_reward_of_agent_one:", self.last_reward_of_agent_one)
        print("last_reward_of_agent_two:", self.last_reward_of_agent_two)
        print("reward:", reward)
        print("state_of_agent_one:", self.state["agent_one"])
        print("state_of_agent_two:", self.state["agent_two"])
        print("reward_one:", self.state["reward_one"])
        print("reward_two:", self.state["reward_two"])
        print("state_one:", self.state["state_one"])
        print("state_two:", self.state["state_two"])
        print("meta_reward:", self.state["meta_reward"])
        print("meta_action:", self.state["meta_action"])

        #if not self.render_mode is None:

        return (
            self.state,
            reward,
            active_terminated and inactive_terminated,
            active_truncated or inactive_truncated,
            info
        )

    def reset(self, seed=None, options=None):
        # reset state of subagents
        # only one return value because DummyVecEnv only returns one observation
        self.last_state_of_agent_one = self.trained_agent_one.env.reset()
        self.last_state_of_agent_two = self.trained_agent_two.env.reset()
        self.total_reward_of_agent_one = 0.0
        self.total_reward_of_agent_two = 0.0

        # reset subagent selection sequence
        self.switch_sequence = []

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

        # reset parameters for choice-based reward
        self.last_reward_of_agent_one = 0.0
        self.last_reward_of_agent_two = 0.0

        # reset counter
        self.episode_counter += 1
        self.step_counter = 0
        self.counter_without_switch = 0
        #self.last_counter_without_switch = 0
        self.last_action = 0

        return self.state, {}
