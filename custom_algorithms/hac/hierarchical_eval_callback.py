from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, EventCallback
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import gym
import numpy as np
import os
import warnings
from util.custom_evaluation import evaluate_policy
from custom_algorithms.hac.hiearchical_evaluation import evaluate_hierarchical_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.logger import Logger
from collections import deque
import time
from custom_algorithms.hac.hac import HAC
from stable_baselines3.common.utils import safe_mean
from custom_algorithms.hac.hierarchical_env import get_h_envs_from_env
import matplotlib.pyplot as plt
import cv2

class HierarchicalEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param deterministic: Whether to render or not the environment during evaluation
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        deterministic: bool = True,
        verbose: int = 1,
        early_stop_data_column: str = 'test/success_rate',
        early_stop_threshold: float = 1.0,
        early_stop_last_n: int = 5,
        top_level_layer=None,
        logger: Logger = None
    ):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_early_stop_val = -np.inf
        self.deterministic = deterministic
        self.eval_histories = {}
        self.early_stop_data_column = early_stop_data_column
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_last_n = early_stop_last_n
        self.top_level_layer = top_level_layer
        self.logger = logger

        layer_envs = get_h_envs_from_env(eval_env, top_level_layer.time_scales, is_testing_env=True, layer_alg=top_level_layer)
        for idx, eval_env in enumerate(layer_envs):
            # Convert to VecEnv for consistency
            eval_env = top_level_layer._wrap_env(eval_env)
            if isinstance(eval_env, VecEnv):
                assert eval_env.num_envs == 1, "You must pass only one environment for evaluation"
            layer_envs[idx] = eval_env

        self.eval_env = layer_envs[0]
        self.eval_env_layers = layer_envs
        self.log_path = log_path
        self.best_model_save_path = None
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        self.vid_size = 1024, 768
        self.vid_fps = 25
        self.eval_count = 0

    def _on_step(self, log_prefix='') -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            info_list = evaluate_hierarchical_policy(
                self.top_level_layer,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes
            )
            for k,v in info_list.items():
                new_k = k
                if len(v) == 0 or type(v[0]) == bool:
                    continue
                mean = np.mean(v)
                std = np.std(v)
                self.logger.record(new_k + '', mean)
                self.logger.record(new_k + '_std', std)
                if k not in self.eval_histories.keys():
                    self.eval_histories[new_k] = []
                self.eval_histories[new_k].append(mean)

            if self.top_level_layer is not None:
                self.top_level_layer._dump_logs()
                self.logger.info("Log path: {}".format(self.log_path))
                if self.early_stop_data_column in self.eval_histories.keys():
                    if self.eval_histories[self.early_stop_data_column][-1] >= self.best_early_stop_val:
                        self.best_early_stop_val = self.eval_histories[self.early_stop_data_column][-1]
                        if self.log_path is not None:
                            self.top_level_layer.save(os.path.join(self.log_path, "best_model"))
                            self.logger.info("New best mean {}: {:.5f}!".format(self.early_stop_data_column,
                                                                     self.best_early_stop_val))
                            self.logger.info(f"Saving new best model at {self.log_path}/best_model")

                    if len(self.eval_histories[self.early_stop_data_column]) >= self.early_stop_last_n:
                        mean_val = np.mean(self.eval_histories[self.early_stop_data_column][-self.early_stop_last_n:])
                        if mean_val >= self.early_stop_threshold:
                            self.logger.info(
                                "Early stop threshold for {} met: Average over last {} evaluations is {:5f} and threshold is {}. Stopping training.".format(
                                    self.early_stop_data_column, self.early_stop_last_n, mean_val,
                                    self.early_stop_threshold))
                            if self.log_path is not None:
                                self.model.save(os.path.join(self.log_path, "early_stop_model"))
                            return False
                else:
                    self.logger.warn("Warning, early stop data column {} not in eval history keys {}. This should only happen once during initialization.".format(self.early_stop_data_column, self.eval_histories.keys()))
            self.eval_count += 1

