import copy
import os
import sys
from typing import List, Dict
import torch
import gymnasium as gym
import numpy as np
import yaml
from matplotlib import pyplot as plt
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

# FIXME: needed for rendering rgb array
np.set_printoptions(threshold=sys.maxsize)

from src.custom_algorithms.cleanppofm.cleanppofm import CLEANPPOFM
from src.custom_algorithms.cleanppofm.utils import get_summed_up_reward_of_env_or_fm_with_predicted_states_of_fm, \
    get_position_and_object_positions_of_observation, get_observation_of_position_and_object_positions, \
    get_next_position_observation_moonlander

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# HUMANS MINIMAL SWITCHING TIME IS 0.462424S WHICH ARE ~ 5 FRAMES/STEP

class MetaEnvPretrained(gym.Env):
    render_mode = None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self,dodge_best_model_name: str, collect_best_model_name: str,
                 dodge_list_of_object_dict_lists: List[Dict] = None,
                 collect_list_of_object_dict_lists: List[Dict] = None, render_mode=None,
                 with_SoC_in_reward: bool = True, with_SoC_in_observation: bool = True,
                 use_prediction_error: bool = True, use_difficulty: bool = True,
                 can_only_switch_as_often_as_humans: bool = False):
        self.ROOT_DIR = "."
        config_path_dodge_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)), "standard_config.yaml")
        config_path_collect_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                     "standard_config_second_task.yaml")

        with open(config_path_dodge_asteroids, "r") as file:
            config_dodge_asteroids = yaml.safe_load(file)
        with open(config_path_collect_asteroids, "r") as file:
            config_collect_asteroids = yaml.safe_load(file)

        # test if configurations are the same
        config_dodge_asteroids_copy = config_dodge_asteroids.copy()
        config_dodge_asteroids_copy["world"]["objects"].pop("type")
        config_collect_asteroids_copy = config_collect_asteroids.copy()
        config_collect_asteroids_copy["world"]["objects"].pop("type")

        if not config_dodge_asteroids_copy == config_collect_asteroids_copy:
            raise ValueError("Configurations are not the same")

        agent_config = config_dodge_asteroids["agent"]
        world_config = config_dodge_asteroids["world"]

        ### ACTION SPACE ###
        # one action to decide which task to control
        # action 0 --> task 0 can be controlled
        # action 1 --> task 1 can be controlled
        self.action_space = gym.spaces.Discrete(2)

        ### OBSERVATION SPACE ###
        # 10x12 or 30x42 grid for each task
        self.y_position_of_agent = agent_config["size"]
        self.following_observations_size = min(
            agent_config["observation_height"],
            int(world_config["y_height"] - self.y_position_of_agent + 1),
        )
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=2,
        #     shape=(
        #         # self.following_observations_size * (world_config["x_width"] + 2) * 2,
        #         # (self.following_observations_size * (world_config["x_width"] + 2) * 2) + 6,
        #         6,
        #     ),
        #     dtype=np.int64,
        # )
        self.observation_space = gym.spaces.Dict(
            {"image": gym.spaces.Box(low=-10, high=5,
                                     shape=(self.following_observations_size * (world_config["x_width"] + 2) * 2,),
                                     dtype=np.int64),
             "reward_dodge": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
             "reward_collect": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
             "task_switching_costs": gym.spaces.Box(low=0, high=0.5, shape=(1,), dtype=np.float32),
             "task_action": gym.spaces.Box(low=0, high=2, shape=(1,), dtype=np.int64),
             "meta_action": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64),
             "crashed_objects": gym.spaces.Box(low=0, high=74, shape=(1,), dtype=np.int64),
             "collected_objects": gym.spaces.Box(low=0, high=30, shape=(1,), dtype=np.int64)
             }
        )
        self.with_SoC_in_observation = with_SoC_in_observation
        # FIXME: this doesn't work for old trained models, because the sorting is different
        # old version --> alphabetically, SoC_collect, SoC_dodge at the end
        # when adding SoC_dodge and SoC_collect later, they are added at the beginning (SoC_collect, SoC_dodge)
        # quickfix -> put SoCs hardcoded in the observation space above
        if self.with_SoC_in_observation:
            self.observation_space["SoC_dodge"] = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            self.observation_space["SoC_collect"] = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.with_SoC_in_reward = with_SoC_in_reward
        self.use_prediction_error = use_prediction_error
        self.use_difficulty = use_difficulty
        self.can_only_switch_as_often_as_humans = can_only_switch_as_often_as_humans

        # logger
        tmp_path = "/tmp/sb3_log/"
        self.logger = configure(tmp_path, ["stdout", "csv"])
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Load the trained agents
        # FIXME: this is an ugly hack to load the trained agents
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             f"../../../policies/{dodge_best_model_name}"), "rb"
                # f"/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/policies/{dodge_best_model_name}", "rb"
        ) as file:
            print("start loading agents", file)
            self.trained_dodge_asteroids = CLEANPPOFM.load(path=file,
                                                           env=make_vec_env("MoonlanderWorld-dodge-gaussian-v0",
                                                                            n_envs=1))
            self.trained_dodge_asteroids.set_logger(logger=self.logger)
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             f"../../../policies/{collect_best_model_name}"), "rb"
                # f"/home/annika/coding_projects/Scilab-RL-github/Scilab-RL/policies/{collect_best_model_name}", "rb"
        ) as file:
            # same model cannot be loaded twice -> copy does also not work
            self.trained_collect_asteroids = CLEANPPOFM.load(path=file,
                                                             env=make_vec_env("MoonlanderWorld-collect-gaussian-v0",
                                                                              n_envs=1))
            self.trained_collect_asteroids.set_logger(logger=self.logger)
            print("finish loading agents")

        # test if models used the same configuration during training
        if not (
                self.trained_dodge_asteroids.position_predicting == self.trained_collect_asteroids.position_predicting
                and self.trained_dodge_asteroids.reward_predicting == self.trained_collect_asteroids.reward_predicting
                and self.trained_dodge_asteroids.number_of_future_steps == self.trained_collect_asteroids.number_of_future_steps
                and self.trained_dodge_asteroids.fm_trained_with_input_noise == self.trained_collect_asteroids.fm_trained_with_input_noise
                and self.trained_dodge_asteroids.input_noise_on == self.trained_collect_asteroids.input_noise_on
                and self.trained_dodge_asteroids.maximum_number_of_objects == self.trained_collect_asteroids.maximum_number_of_objects):
            raise ValueError("Models used different configurations during training")

        # because dodge and collect use the same configuration, we can use one of them
        self.observation_height = self.trained_dodge_asteroids.env.env_method("get_wrapper_attr", "observation_height")[
            0]
        self.observation_width = self.trained_dodge_asteroids.env.env_method("get_wrapper_attr", "observation_width")[0]
        self.agent_size = self.trained_dodge_asteroids.env.env_method("get_wrapper_attr", "size")[0]
        self.maximum_number_of_objects = self.trained_dodge_asteroids.maximum_number_of_objects

        # the state could possibly be a belief state of the forward model
        # only one return value because DummyVecEnv only returns one observation
        self.state_of_dodge_asteroids = self.trained_dodge_asteroids.env.reset()
        self.state_of_collect_asteroids = self.trained_collect_asteroids.env.reset()

        # concatenate states (possibly belief states)
        self.current_task = 0
        state_image = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(self.observation_height, self.observation_width + 2),
             self.state_of_collect_asteroids.reshape(self.observation_height, self.observation_width + 2)),
            axis=1,
        ).flatten()

        self.SoC_dodge = 1
        self.SoC_collect = 1

        # add SoC_dodge, SoC_collect, reward_dodge, reward_collect, task_action, meta_action
        # self.state = np.append(self.state, [self.SoC_dodge, self.SoC_collect, 0, 0, 0, 0])
        # self.state = np.array([self.SoC_dodge, self.SoC_collect, 0, 0, 0, 0])
        self.state = {"image": state_image, "reward_dodge": 0, "reward_collect": 0, "task_switching_costs": 0,
                      "task_action": 0, "meta_action": 0, "crashed_objects": 0, "collected_objects": 0}
        if self.with_SoC_in_observation:
            self.state["SoC_dodge"] = self.SoC_dodge
            self.state["SoC_collect"] = self.SoC_collect

        # make subagents use normalized rewards

        # for rendering
        plt.ion()
        self.fig, self.ax = plt.subplots()
        eximg = np.zeros((self.observation_height, self.observation_width * 2 + 4))
        eximg[0] = -10
        eximg[1] = 5
        self.im = self.ax.imshow(eximg)

        # counter
        self.episode_counter = 0
        self.step_counter = 0
        self.counter_without_switch = 0
        self.last_action = 0

        # when using benchmark, the object positions are predefined and should be set in moonlander env
        self.dodge_list_of_object_dict_lists = dodge_list_of_object_dict_lists
        self.collect_list_of_object_dict_lists = collect_list_of_object_dict_lists
        if self.dodge_list_of_object_dict_lists is not None and self.episode_counter < len(
                self.dodge_list_of_object_dict_lists):
            self.trained_dodge_asteroids.env.env_method("set_object_dict_list",
                                                        self.dodge_list_of_object_dict_lists[self.episode_counter])
        if self.collect_list_of_object_dict_lists is not None and self.episode_counter < len(
                self.collect_list_of_object_dict_lists):
            self.trained_collect_asteroids.env.env_method("set_object_dict_list",
                                                          self.collect_list_of_object_dict_lists[self.episode_counter])

    def step(self, action: int):
        # task switch
        if self.last_action == action:
            self.counter_without_switch += 1
            task_switch = False
        else:
            self.counter_without_switch = 0
            self.last_action = action
            task_switch = True

        if not self.can_only_switch_as_often_as_humans:
            state, reward, is_done, truncated, info = self.actual_step_logic(action=action,
                                                                             task_switch=task_switch)
        else:
            # restrict switching to at least 5 frames
            if task_switch:
                # do 5 steps with the same action
                state, reward, is_done, truncated, info = self.actual_step_logic(action=action, task_switch=task_switch)
                for _ in range(4):
                    # FIXME: all stuff is just overwritten -> is this a good solution?
                    state, reward, is_done, truncated, info = self.actual_step_logic(action=action, task_switch=False)
            else:
                state, reward, is_done, truncated, info = self.actual_step_logic(action=action, task_switch=task_switch)

        return state, reward, is_done, truncated, info

    def actual_step_logic(self, action: int, task_switch: bool):
        """
        action: selects the task
                0: one
                1: two
        """
        # action 0: dodge asteroids
        # action 1: collect asteroids
        match action:

            case 0:
                # dodge task
                active_model = self.trained_dodge_asteroids
                active_last_state = self.state_of_dodge_asteroids
                inactive_model = self.trained_collect_asteroids
                inactive_last_state = self.state_of_collect_asteroids
                inactive_SoC = self.SoC_collect
                self.current_task = 0
            case 1:
                # collect task
                active_model = self.trained_collect_asteroids
                active_last_state = self.state_of_collect_asteroids
                inactive_model = self.trained_dodge_asteroids
                inactive_last_state = self.state_of_dodge_asteroids
                inactive_SoC = self.SoC_dodge
                self.current_task = 1

            # If an exact match is not confirmed, this last case will be used if provided
            case _:
                raise ValueError("action must be 0, 1")

        ### ACTIVE TASK ###
        # predict next action
        action_of_task_agent, _, _ = active_model.predict(active_last_state, deterministic=True)
        # get position and object positions of observation
        active_agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(
            torch.tensor(active_last_state, device=device),
            observation_width=self.observation_width,
            observation_height=self.observation_height,
            maximum_number_of_objects=active_model.maximum_number_of_objects,
            agent_size=self.agent_size)
        # forward model predictions once with state and action to get next agent and object positions
        # active_belief_state_normal_distribution = active_model.fm_network(active_agent_and_object_positions_tensor,
        #                                                                   torch.tensor([action_of_task_agent]).float())
        # FIXME: hardcoded forward model prediction
        active_gold_label = get_next_position_observation_moonlander(
            observations=active_agent_and_object_positions_tensor,
            actions=action_of_task_agent,
            observation_width=self.observation_width,
            observation_height=self.observation_height,
            agent_size=self.agent_size,
            maximum_number_of_objects=self.maximum_number_of_objects)
        # form to normal distribution
        active_gold_label = torch.distributions.Normal(active_gold_label,
                                                       scale=torch.tensor(
                                                           [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                             1., 1., 1., 1., 1., 1., 1., 1.]]))
        # perform action & SoC calculation & reward estimation corrected by SoC
        (new_state, _, active_is_done, active_info, active_prediction_error, active_difficulty, active_SoC,
         active_normalized_reward_estimation_corrected_by_SoC, input_noise,
         active_normalized_reward) = active_model.step_in_env(
            actions=torch.tensor(action_of_task_agent).float(),
            # forward_normal=active_belief_state_normal_distribution)
            forward_normal=active_gold_label,
            use_reward_of_env=True, use_prediction_error=self.use_prediction_error, use_difficulty=self.use_difficulty)

        ### INACTIVE TASK ###
        # set input noise to zero
        inactive_model.env.env_method("set_input_noise", 0)
        # perform default action 1 in inactive task
        # only four return value because DummyVecEnv only returns observation, reward, done, info
        # but meta agent does not see actual state and reward
        _, _, inactive_is_done, inactive_info = inactive_model.env.step(torch.tensor([1], device=device))
        # get position and object positions of observation
        inactive_agent_and_object_positions_tensor = get_position_and_object_positions_of_observation(
            torch.tensor(inactive_last_state, device=device),
            observation_width=self.observation_width,
            observation_height=self.observation_height,
            maximum_number_of_objects=inactive_model.maximum_number_of_objects,
            agent_size=self.agent_size)
        # forward model predictions once with state and action to get next belief state
        # inactive_belief_state_normal_distribution = inactive_model.fm_network(
        #     inactive_agent_and_object_positions_tensor,
        #     torch.tensor([[1]]).float())
        # FIXME: hardcoded forward model prediction
        inactive_gold_label = get_next_position_observation_moonlander(
            observations=inactive_agent_and_object_positions_tensor,
            actions=torch.tensor([1]),
            observation_width=self.observation_width,
            observation_height=self.observation_height,
            agent_size=self.agent_size,
            maximum_number_of_objects=self.maximum_number_of_objects)
        # form to normal distribution
        inactive_gold_label = torch.distributions.Normal(inactive_gold_label,
                                                         scale=torch.tensor(
                                                             [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                               1., 1., 1., 1., 1., 1., 1., 1.]]))
        # get new inactive state from forward model position prediction
        belief_state = get_observation_of_position_and_object_positions(agent_and_object_positions=
                                                                        # inactive_belief_state_normal_distribution.mean[
                                                                        #     0][:-1].cpu().unsqueeze(0),
                                                                        inactive_gold_label.mean,
                                                                        observation_height=self.observation_height,
                                                                        observation_width=self.observation_width,
                                                                        agent_size=self.agent_size,
                                                                        task=inactive_model.env.env_method(
                                                                            "get_wrapper_attr", "task")[
                                                                            0]).flatten().cpu().numpy()
        belief_state = np.expand_dims(belief_state, 0)
        # SoC update --> degrade SoC by factor of observation heigth, so that after half of the steps of the observation
        # the SoC is 0.5 and after all steps the SoC is 0
        inactive_SoC = min(max(0, inactive_SoC - (1 / self.observation_height)), 1)
        # simulate future n steps
        # FIXME: for now it is hardcoded 5 steps + can be deleted in cleanppofm?
        # reward estimation -> predict next state -> get reward of environment
        inactive_summed_up_rewards = get_summed_up_reward_of_env_or_fm_with_predicted_states_of_fm(
            env=inactive_model.env,
            fm_network=inactive_model.fm_network,
            last_observation=
            # inactive_belief_state_normal_distribution.mean[
            #     0][:-1].unsqueeze(0),
            inactive_gold_label.mean,
            reward_from_env=True,
            env_name="MoonlanderWorldEnv",
            position_predicting=True,
            number_of_future_steps=int(self.observation_height / 2),
            maximum_number_of_objects=inactive_model.maximum_number_of_objects)

        # FIXME: put in?
        # not needed because already introduced by inactive SoC
        # inactive_summed_up_rewards =
        # min(max(0, inactive_summed_up_rewards - (self.counter_without_switch * (1/self.observation_height))), 1)
        # reward estimation corrected by SoC
        inactive_reward_estimation_corrected_by_SoC = (inactive_summed_up_rewards + inactive_SoC) / 2

        match action:
            case 0:
                # dodge task
                self.state_of_dodge_asteroids = new_state
                info_dodge = active_info
                if self.with_SoC_in_reward:
                    reward_dodge = active_normalized_reward_estimation_corrected_by_SoC
                else:
                    reward_dodge = active_normalized_reward
                self.SoC_dodge = active_SoC
                self.state_of_collect_asteroids = belief_state
                info_collect = inactive_info
                if self.with_SoC_in_reward:
                    reward_collect = inactive_reward_estimation_corrected_by_SoC
                else:
                    reward_collect = inactive_summed_up_rewards
                self.SoC_collect = inactive_SoC
                # for debugging
                last_dodge_position = int(active_agent_and_object_positions_tensor[0][0])
                last_collect_position = int(inactive_agent_and_object_positions_tensor[0][0])
                dodge_action = action_of_task_agent
                collect_action = 1
                next_dodge_position = int(
                    get_position_and_object_positions_of_observation(torch.tensor(new_state, device=device),
                                                                     observation_width=self.observation_width,
                                                                     observation_height=self.observation_height,
                                                                     maximum_number_of_objects=active_model.maximum_number_of_objects,
                                                                     agent_size=self.agent_size)[0][0])
                next_collect_position = int(
                    get_position_and_object_positions_of_observation(torch.tensor(belief_state, device=device),
                                                                     observation_width=self.observation_width,
                                                                     observation_height=self.observation_height,
                                                                     maximum_number_of_objects=inactive_model.maximum_number_of_objects,
                                                                     agent_size=self.agent_size)[0][0])
                predicted_next_dodge_position = round(
                    # min(max(self.agent_size, active_belief_state_normal_distribution.mean.cpu().detach().numpy()[0][0]),
                    min(max(self.agent_size, active_gold_label.mean.cpu().detach().numpy()[0][0]),
                        self.observation_width - self.agent_size + 1))
                predicted_next_collect_position = round(
                    min(max(self.agent_size,
                            # inactive_belief_state_normal_distribution.mean.cpu().detach().numpy()[0][0]),
                            inactive_gold_label.mean.cpu().detach().numpy()[0][0]),
                        self.observation_width - self.agent_size + 1))
            case 1:
                # collect task
                self.state_of_dodge_asteroids = belief_state
                info_dodge = inactive_info
                if self.with_SoC_in_reward:
                    reward_dodge = inactive_reward_estimation_corrected_by_SoC
                else:
                    reward_dodge = inactive_summed_up_rewards
                self.SoC_dodge = inactive_SoC
                self.state_of_collect_asteroids = new_state
                info_collect = active_info
                if self.with_SoC_in_reward:
                    reward_collect = active_normalized_reward_estimation_corrected_by_SoC
                else:
                    reward_collect = active_normalized_reward
                self.SoC_collect = active_SoC
                # for debugging
                last_dodge_position = int(inactive_agent_and_object_positions_tensor[0][0])
                last_collect_position = int(active_agent_and_object_positions_tensor[0][0])
                dodge_action = 1
                collect_action = action_of_task_agent
                next_dodge_position = int(
                    get_position_and_object_positions_of_observation(torch.tensor(belief_state, device=device),
                                                                     observation_width=self.observation_width,
                                                                     observation_height=self.observation_height,
                                                                     maximum_number_of_objects=inactive_model.maximum_number_of_objects,
                                                                     agent_size=self.agent_size)[0][0])
                next_collect_position = int(
                    get_position_and_object_positions_of_observation(torch.tensor(new_state, device=device),
                                                                     observation_width=self.observation_width,
                                                                     observation_height=self.observation_height,
                                                                     maximum_number_of_objects=active_model.maximum_number_of_objects,
                                                                     agent_size=self.agent_size)[0][0])
                predicted_next_dodge_position = round(
                    # min(max(1, inactive_belief_state_normal_distribution.mean.cpu().detach().numpy()[0][0]), 10))
                    min(max(1, inactive_gold_label.mean.cpu().detach().numpy()[0][0]), 10))
                predicted_next_collect_position = round(
                    # min(max(1, active_belief_state_normal_distribution.mean.cpu().detach().numpy()[0][0]), 10))
                    min(max(1, active_gold_label.mean.cpu().detach().numpy()[0][0]), 10))
            case _:
                raise ValueError("action must be 0, 1")

        state_image = np.concatenate(
            (
                self.state_of_dodge_asteroids.reshape(self.observation_height, self.observation_width + 2),
                self.state_of_collect_asteroids.reshape(self.observation_height, self.observation_width + 2),
            ),
            axis=1,
        ).flatten()

        # numpy array (2520,) + SoC_dodge + SoC_collect + reward_dodge + reward_collect + task_action + meta_action
        # self.state = np.append(self.state,
        #                        [self.SoC_dodge, self.SoC_collect, reward_dodge, reward_collect, action_of_task_agent,
        #                         action])
        # self.state = np.array(
        #     [self.SoC_dodge, self.SoC_collect, reward_dodge, reward_collect, action_of_task_agent, action])
        if task_switch:
            task_switch_costs = 0.5
        else:
            task_switch_costs = 0
        self.state = {"image": state_image, "reward_dodge": reward_dodge, "reward_collect": reward_collect,
                      "task_switching_costs": task_switch_costs, "task_action": action_of_task_agent,
                      "meta_action": action, "crashed_objects": info_dodge[0]["number_of_crashed_or_collected_objects"],
                      "collected_objects": info_collect[0]["number_of_crashed_or_collected_objects"]}
        if self.with_SoC_in_observation:
            self.state["SoC_dodge"] = self.SoC_dodge
            self.state["SoC_collect"] = self.SoC_collect

        self.step_counter += 1
        info = {"info_dodge": info_dodge, "info_collect": info_collect, "reward_dodge": reward_dodge,
                "reward_collect": reward_collect, "task_switching_costs": task_switch_costs, "action_meta": action,
                "dodge_position_before": last_dodge_position,
                "collect_position_before": last_collect_position, "dodge_action": dodge_action,
                "collect_action": collect_action, "input_noise": input_noise,
                "dodge_next_position": next_dodge_position, "collect_next_position": next_collect_position,
                "predicted_dodge_next_position": predicted_next_dodge_position,
                "predicted_collect_next_position": predicted_next_collect_position,
                "prediction_error": active_prediction_error, "difficulty": active_difficulty,
                "SoC_dodge": self.SoC_dodge, "SoC_collect": self.SoC_collect}

        return (
            self.state,
            reward_dodge + reward_collect - task_switch_costs,
            active_is_done or inactive_is_done,
            False,
            info,
        )

    def render(self):
        # observation = copy.deepcopy(self.state).reshape(self.observation_height, self.observation_width * 2 + 4)
        observation = copy.deepcopy(self.state["image"]).reshape(self.observation_height,
                                                                 self.observation_width * 2 + 4)
        # place frame around current task
        if self.current_task == 0:
            first_fill_value = -10
            second_fill_value = -1
            tmp = np.array([-10, -1])
            # +2 for walls + 2 for frame
            row = np.expand_dims(np.repeat(tmp, self.observation_width + 4), axis=0)
        else:
            first_fill_value = -1
            second_fill_value = -10
            tmp = np.array([-1, -10])
            # +2 for walls + 2 for frame
            row = np.expand_dims(np.repeat(tmp, self.observation_width + 4), axis=0)

        observation = np.concatenate(
            (
                np.full((self.observation_height, 1), first_fill_value),
                observation[:, :self.observation_width + 2],
                np.full((self.observation_height, 1), first_fill_value),
                np.full((self.observation_height, 1), second_fill_value),
                observation[:, self.observation_width + 2:],
                np.full((self.observation_height, 1), second_fill_value)),
            axis=1)
        observation = np.concatenate((row, observation, row), axis=0)

        self.im.set_data(observation)
        if self.render_mode == "human":
            self.fig.canvas.draw_idle()
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,))

    def reset(self, seed=None, options=None):
        # the state could possibly be a belief state of the forward model
        # only one return value because DummyVecEnv only returns one observation
        self.state_of_dodge_asteroids = self.trained_dodge_asteroids.env.reset()
        self.state_of_collect_asteroids = self.trained_collect_asteroids.env.reset()

        # concatenate states (possibly belief states)
        self.current_task = 0
        state_image = np.concatenate(
            (self.state_of_dodge_asteroids.reshape(self.observation_height, self.observation_width + 2),
             self.state_of_collect_asteroids.reshape(self.observation_height, self.observation_width + 2)),
            axis=1,
        ).flatten()

        self.SoC_dodge = 1
        self.SoC_collect = 1

        # add SoC_dodge, SoC_collect, reward_dodge, reward_collect, task_action, meta_action
        # self.state = np.append(self.state, [self.SoC_dodge, self.SoC_collect, 0, 0, 0, 0])
        # self.state = np.array([self.SoC_dodge, self.SoC_collect, 0, 0, 0, 0])
        self.state = {"image": state_image, "reward_dodge": 0, "reward_collect": 0, "task_switching_costs": 0,
                      "task_action": 0, "meta_action": 0, "crashed_objects": 0, "collected_objects": 0}
        if self.with_SoC_in_observation:
            self.state["SoC_dodge"] = self.SoC_dodge
            self.state["SoC_collect"] = self.SoC_collect

        # counter
        self.episode_counter += 1
        self.step_counter = 0
        self.counter_without_switch = 0
        self.last_action = 0

        # when using benchmark, the object positions are predefined and should be set in moonlander env
        if self.dodge_list_of_object_dict_lists is not None and self.episode_counter < len(
                self.dodge_list_of_object_dict_lists):
            self.trained_dodge_asteroids.env.env_method("set_object_dict_list",
                                                        self.dodge_list_of_object_dict_lists[self.episode_counter])
        if self.collect_list_of_object_dict_lists is not None and self.episode_counter < len(
                self.collect_list_of_object_dict_lists):
            self.trained_collect_asteroids.env.env_method("set_object_dict_list",
                                                          self.collect_list_of_object_dict_lists[self.episode_counter])
        return self.state, {}