"""
This file is used to test all of the different algorithms possible to use to solve our Host AI, table management software problem.
The full test will take a while to run because it goes through many different algorithms.

Each test instantiates a test object and then releases the object once the trial has been run and saves a bunch of data on how
the model did.

Many of the proposed solutions are combinations of different methods, therefore each of those method-combos has its own trial
BC_PPO_RUDDER

"""
from enum import Enum

from BasicRestaurant1 import MBPost
from analysis.algorithms.CL_PPO_RUDDER import CL_PPO_RUDDER
class TestingEnvironment(object):
    pass





if __name__ == '__main__':
    class args:
        log_dir = "./logs/"
        CL_step = 1
        total_timesteps = 30000
        track_wandb = True
        wandb_project_name = "hostai"
        wandb_entity = None
        ent_coef = 0
        gamma = 0.99
        n_steps = 128
        target_kl = None
        n_epochs = 1
        learning_rate = 0.001
        envlogger = True
        envlogger_freq = 100
        clip_range = 0.2
        track_local = True
        batch_size = 64
        env_steps = 100
        seq_gen = False
        human_player = False
        bc = False
        phase = '0a'
        seed=42
        tensorboard_log = './logs/tensorboard/'

    cl_ppo_rudder = CL_PPO_RUDDER(args,restaurant=MBPost())
    #cl_ppo_rudder.run_phase_0()
    cl_ppo_rudder.run_phase_1()