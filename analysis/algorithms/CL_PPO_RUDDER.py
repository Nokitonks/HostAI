"""
This algorithm uses a curriculum learning style approach to help out basis PPO model learn
Further, reward shaping is used in the curriculum process in order to redistribute rewards to positive sequences of actions

- Ideally the reward should propagate backwards to every action that caused a specific reward signal s/t when the final
    model gets to the original delayed reward, it no longer collects a reward at that step

- There are multiple methods that are used to transfer learning between different tasks. Here we are going to initialize our
next model with the previous best model. In addition, for each sub-task we are going to run LSTM-RUDDER to redistribute
rewards to specific state-action pairs

--------------
Phase 0
--------------

Our goal for this section is to have the model the most basic parts of our environment. For this phase, the only available
actions will be assign_party actions. Essentially this is a puzzle where the model must seat parties at the correct size in order to gain the most reward

There will always be enough space in the restaurant (without combining tables) to fit every single party
space to fit all the reservations if the model sits people at tables that align with their size.

We utilize RUDDER here to shape the rewards such that the earliest "action" in the sequence -> having a table available
of for size of reservation at when the reservation is going to arrive. Final reward given at time that the table leaves the restaurant.

These RUDDER sequences are collected only when the model has surpassed a certain score for the episode. The redistributed rewards are
not applied until the next phase

The model should ideally learn that sitting parties at different tables has the same expected reward.

Once the model is able to sit every party at the full reservation size for a variety of different arrival times we can move to phase 1

--------------
Phase 2
--------------
"""
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from BasicRestaurant1 import MBPost
from HostEnv import HostWorldEnv, mask_fn
from utils.callbacks import CL_PPO_RUDDER_PHASE_0_Callback


class CL_PPO_RUDDER(object):

    def __init__(self, args,restaurant):
        self.args = args
        self.phase = 0
        self.restaurant = restaurant
        immutable_settings = restaurant.immutable_settings

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
        env = HostWorldEnv(immutable_config=immutable_settings, mutable_config=default_mutable_settings)
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
                        n_steps=args.n_steps,
                        tensorboard_log=args.tensorboard_log,
                        target_kl=args.target_kl,
                        n_epochs=args.n_epochs,
                        learning_rate=args.learning_rate,
                        max_grad_norm=0.5,
                        )


    def run_phase_0(self):
        info = {}
        info['tables'] = self.env.tables
        info['train_stage_episodes'] = 100
        phase_0_callback = CL_PPO_RUDDER_PHASE_0_Callback(True,1,info,1)
        self.model.learn(total_timesteps=100000000, callback=[phase_0_callback], progress_bar=True)
        pass