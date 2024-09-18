"""
This algorithm uses a curriculum learning style approach to help out basis PPO model learn
Further, reward shaping is used in the curriculum process in order to redistribute rewards to positive sequences of actions

- Ideally the reward should propagate backwards to every action that caused a specific reward signal s/t when the final
    model gets to the original delayed reward, it no longer collects a reward at that step

- There are multiple methods that are used to transfer learning between different tasks. Here we are going to initialize our
next model with the previous best model. In addition, for each sub-task we are going to run LSTM-RUDDER to redistribute
rewards to specific state-action pairs

--------------
Phase 0a + 0b + 0c
--------------

Our goal for this section is to have the model the most basic parts of our environment. For this phase, the only available
actions will be assign_party actions. Essentially this is a puzzle where the model must seat parties at the correct size in order to gain the most reward

There will always be enough space in the restaurant (without combining tables) to fit every single party
space to fit all the reservations if the model sits people at tables that align with their size.

The model should ideally learn that sitting parties at different tables has the same expected reward.

Once the model is able to sit every party at the full reservation size for a variety of different arrival times we can move to phase 1

phase_0a and phase_0b differentiate learning the assign_reservation actions and the assign_walk_in actions and then
phase_0c is a mix between the two.

--------------
Phase 2
--------------

We are now going to introduce time dependency into our model this means the model will be able to use the advance time action.
In addition, we remove rewards from the initial assign action and only give the reward once the table has left. Hopefully the
model should still be able to do resonably well because it has learned to use the assign actions. Should converge to the max
value again.

Parties will come in from a mix of walk-ins and reservations

Most importantly in this step we are learning to train our LSTM such that it can redistribute the rewards backwards in time
to initial actions

** Lets think **
Do we actually need to train our LSTM with this simplistic data? Maybe. Will likely need to see if it can learn quick enough
at the complex task level to see if it needs this initial training or not. TBD






"""
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import wandb
from BasicRestaurant1 import MBPost
from HostEnv import HostWorldEnv, mask_fn
from utils.callbacks import CL_PPO_RUDDER_PHASE_0_Callback, EnvLogger, CL_PPO_RUDDER_PHASE_1_Callback
from wandb.integration.sb3 import WandbCallback
import torch

from utils.helperFunctions import clear_csv_file_keep_headers


class CL_PPO_RUDDER(object):

    def __init__(self, args,restaurant):
        self.args = args
        self.phase = 0
        self.restaurant = restaurant
        self.immutable_settings = restaurant.immutable_settings

        """
        Below we have the settings for the first phase
        """
        default_mutable_settings = {
            "clean_time": {2: 1,
                           4: 5,
                           6: 5,
                           8: 5
                           },
            "wait_tolerance": 100,
            "reservations_path": 'reservation_files/cl_ppo_rudder/phase_0.csv',
            "log_dir": args.log_dir,
            "end_time": 2,
            'walk_ins_path': 'walk_in_files/cl_ppo_rudder/phase_0.csv',
            'num_servers':1,
            'server_sections':{},
            'phase': args.phase
        }
        #Every table is 0 servers section
        for table in restaurant.tables:
            table_number = table.number
            default_mutable_settings['server_sections'][str(table_number)] = 0
        self.mutable_settings = default_mutable_settings
        env = HostWorldEnv(immutable_config=self.immutable_settings, mutable_config=default_mutable_settings)
        env = FlattenObservation(env)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        env = Monitor(env, default_mutable_settings['log_dir'])
        self.env = env
        self.model = MaskablePPO("MlpPolicy",
                        env,
                        seed=args.seed,
                        verbose=1,
                        ent_coef=args.ent_coef,
                        gamma=args.gamma,
                        tensorboard_log=args.tensorboard_log,
                        n_steps=args.n_steps,
                        target_kl=args.target_kl,
                        n_epochs=args.n_epochs,
                        learning_rate=args.learning_rate,
                        max_grad_norm=0.5,
                        )


    def run_phase_0(self):
        info = {}
        info['tables'] = self.env.tables
        info['train_stage_episodes'] = 100
        config = {
            "policy_type": "MlpPolicy",
            'total_timesteps': self.args.total_timesteps,

            "policy_kwargs": dict(
                activation_fn=torch.nn.Sigmoid),
            "batch_size": self.args.batch_size,
        }
        if self.args.track_wandb:
            run = wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=True,
                # config=vars(args),
                config=config,
                name="cl_ppo_rudder_phase_0a",
                monitor_gym=True,
                save_code=True,
                reinit=True
            )

            # log the code
            wandb.run.log_code(".")

            # wandb callback
            wandbc = WandbCallback(gradient_save_freq=2000, model_save_path=f"models/{run.id}",
                                   verbose=0)
        else:
            wandbc = None

        phase_0_callback = CL_PPO_RUDDER_PHASE_0_Callback(True,20,info,'a',122)
        env_logger = EnvLogger(self.args.envlogger_freq, './logs/cl_ppo_rudder/statevar/', self.args.seq_gen, None)

        ## Teach Phase 0a ###
        self.model.learn(total_timesteps=50000, callback=[phase_0_callback,env_logger,wandbc], progress_bar=True)
        #Will reach here once model has gotten 100% with the reservation sittings
        self.model.save('models/cl_ppo_rudder/phase_0a')

        clear_csv_file_keep_headers('reservation_files/cl_ppo_rudder/phase_0.csv')
        phase_0_callback = CL_PPO_RUDDER_PHASE_0_Callback(True,20,info,'b',122)

        self.env.mutable_config['phase'] = "0b"
        self.env.reset()
        ## Teach Phase 0b ###
        self.model.learn(total_timesteps=50000, callback=[phase_0_callback,env_logger,wandbc], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_0b')

        phase_0_callback = CL_PPO_RUDDER_PHASE_0_Callback(True,20,info,'c',122)
        self.env.mutable_config['phase'] = "0c"
        self.env.reset()
        ## Teach Phase 0b ###
        self.model.learn(total_timesteps=50000, callback=[phase_0_callback,env_logger,wandbc], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_0c')

        return True

    def run_phase_1(self):

        info = {}
        info['tables'] = self.env.tables
        self.mutable_settings['reservations_path'] = 'reservation_files/cl_ppo_rudder/phase_1.csv'
        self.mutable_settings['walk_ins_path'] = 'walk_in_files/cl_ppo_rudder/phase_1.csv'
        self.mutable_settings['end_time'] = 20
        self.mutable_settings['phase'] = '1a'

        env = HostWorldEnv(immutable_config=self.immutable_settings, mutable_config=self.mutable_settings)
        env = FlattenObservation(env)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        env = Monitor(env, self.mutable_settings['log_dir'])
        self.env = env
        self.env.reset()
        self.model = MaskablePPO.load('models/cl_ppo_rudder/phase_0c', env=self.env)
        phase_1_callback = CL_PPO_RUDDER_PHASE_1_Callback(True,20,info,'a',122)
        env_logger = EnvLogger(self.args.envlogger_freq, './logs/cl_ppo_rudder/statevar/', self.args.seq_gen, None)

        ## Teach Phase 0a ###
        self.model.learn(total_timesteps=100000, callback=[phase_1_callback,env_logger], progress_bar=True)
        #Will reach here once model has gotten 100% with the reservation sittings
        self.model.save('models/cl_ppo_rudder/phase_1a')
        pass