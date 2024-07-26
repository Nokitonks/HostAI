import torch

from HostEnv import HostWorldEnv
import pygame
import time
from ClassDefinitions import *
from MBPost import MBPPOST_TABLES, SMALL_TABLES
import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
from HostAgent import HostAgent
from gymnasium.wrappers import FlattenObservation
import tensorflow as tf
from stable_baselines3 import A2C, DQN
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_action_mask()


# Train using StableBaseline3. Lots of hardcoding for simplicity i.e. use of the A2C (Advantage Actor Critic) algorithm.
def train_sb3(env):
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = MaskablePPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)  # train
        model.save(f"{model_dir}/Maskable_{TIMESTEPS * iters}")  # Save a trained model every TIMESTEPS

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(env,render=True):

    # Load model
    model = MaskablePPO.load('models/Maskable_90000', env=env)
    #print(evaluate_policy(model, env, n_eval_episodes=20))
    total = 0
    for episode in range(20):
        # Run a test
        obs = env.reset()[0]
        terminated = False
        score = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            actions_masks = env.get_action_mask()
            action, _ = model.predict(observation=obs,action_masks=actions_masks) # Turn on deterministic, so predict always returns the same behavior
            obs, reward, terminated, _, _ = env.step(action.item())
            env.render()
            time.sleep(0.55)
            score += reward
            if terminated:
                print(f'score {score}')
                total += score
                break
    print(total/20)

if __name__ == "__main__":

    config = {
        'level_settings' : LevelSettings(SMALL_TABLES,8,220,30,30,
                                        ((SMALL_TABLES[0],SMALL_TABLES[1]),
                                         (SMALL_TABLES[0], SMALL_TABLES[2]),
                                         (SMALL_TABLES[0],SMALL_TABLES[3]),
                                         (SMALL_TABLES[1], SMALL_TABLES[3]),
                                         (SMALL_TABLES[2],SMALL_TABLES[3]),
                                         (SMALL_TABLES[1],SMALL_TABLES[2]))
                                         ),
        'max_party_size' : 8,
        'max_time': 220,
        'max_wait_list': 30,
        'max_reservation_list' : 30,
        'window_size' : (800, 600),
        'grid_size' : 25,
        'wait_tolerance': 50
    }
    env = HostWorldEnv(config)

    episodes = 20
    env = FlattenObservation(env)
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    # train_sb3(env)
    test_sb3(env)

    for episode in tqdm(range(episodes)):
        state, info = env.reset()
        done = False
        score = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # Use the action mask to filter valid actions
            action_mask = env.get_action_mask()

            # Check if all values are False by directly passing the list to `all`
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            action = env.action_space.sample(action_mask)
            if not any(action_mask):
                action = None
            else:
                while not action_mask[action]:
                    action = env.action_space.sample(action_mask)

            observation, reward, done, trunkated, info = env.step(action)

            state = observation
            score += reward

            # update the agent
            env.render()
        print(f"Episode{episode}: score {score}")
    env.close()