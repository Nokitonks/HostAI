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
Phase 1
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

We want each phase to push back the eventual reward of people sat/served to all the little things during the evening
that affect that final outcome



--------------
Phase 2
--------------

Here we begin our introduction into combining tables. This is perhaps the craziest phase. We are going to go back to a
time invariant environment





"""
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import wandb

from BasicHost import BasicHost
from BasicRestaurant1 import MBPost
from HostEnv import HostWorldEnv, mask_fn
from rudder import LessonBuffer, RRLSTM
from utils.callbacks import CL_PPO_RUDDER_PHASE_0_Callback, EnvLogger, CL_PPO_RUDDER_PHASE_1_Callback, RudderManager, \
    CL_PPO_RUDDER_PHASE_2_Callback, SaveOnBestTrainingRewardCallback, CL_PPO_RUDDER_PHASE_3_Callback, \
    CL_PPO_RUDDER_PHASE_4_Callback
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
        ## Teach Phase 0c ###
        self.model.learn(total_timesteps=50000, callback=[phase_0_callback,env_logger,wandbc], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_0c')

        return True

    def run_phase_1(self):

        info = {}
        info['tables'] = self.env.tables
        self.mutable_settings['reservations_path'] = 'reservation_files/cl_ppo_rudder/phase_1.csv'
        self.mutable_settings['walk_ins_path'] = 'walk_in_files/cl_ppo_rudder/phase_1.csv'
        self.mutable_settings['end_time'] = 80
        self.mutable_settings['n_steps'] = 90
        self.mutable_settings['phase'] = '1a'


        env = HostWorldEnv(immutable_config=self.immutable_settings, mutable_config=self.mutable_settings)
        env = FlattenObservation(env)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        env = Monitor(env, self.mutable_settings['log_dir'])

        # LSTM
        lb_size = 264
        n_lstm = 264
        n_steps = 80
        policy_lr = 0.1
        lstm_lr = 1e-2
        l2_regularization = 1e-6
        avg_window = 750
        print(f"Action size = {env.get_n_actions()[-1]}, State size = {env.get_state_shape()[-1]}\n")

        self.env = env
        self.env.reset()
        self.model = MaskablePPO.load('models/cl_ppo_rudder/phase_0c', env=self.env)
        phase_1_callback = CL_PPO_RUDDER_PHASE_1_Callback(True,10,info,'a',122)
        env_logger = EnvLogger(self.args.envlogger_freq, './logs/cl_ppo_rudder/statevar/', self.args.seq_gen, [])

        ## Teach Phase 1a ###
        self.model.learn(total_timesteps=100000, callback=[phase_1_callback,env_logger], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_1a')

    def run_phase_2(self):


        info = {}
        info['tables'] = self.env.tables
        self.mutable_settings['reservations_path'] = 'reservation_files/cl_ppo_rudder/phase_2.csv'
        self.mutable_settings['walk_ins_path'] = 'walk_in_files/cl_ppo_rudder/phase_2.csv'
        self.mutable_settings['end_time'] = 2
        self.mutable_settings['n_steps'] = 60
        self.mutable_settings['phase'] = '2a'

        env = HostWorldEnv(immutable_config=self.immutable_settings, mutable_config=self.mutable_settings)
        env = FlattenObservation(env)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        env = Monitor(env, self.mutable_settings['log_dir'])
        self.env = env
        self.env.reset()
        # LSTM
        lb_size = 512
        n_lstm = 32
        n_steps = 60
        policy_lr = 0.1
        lstm_lr = 1e-2
        l2_regularization = 1e-6
        avg_window = 750
        scaling_factor = 0.11

        lesson_buffer = LessonBuffer(size=lb_size, max_time=n_steps, n_features=env.get_state_shape()[-1])

        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.args.log_dir)
        rudder_lstm = RRLSTM(state_input_size=0, n_actions=env.get_n_actions()[-1],
                             buffer=lesson_buffer, n_units=n_lstm,
                             lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                             lstm_batch_size=128, continuous_pred_factor=0.5)
        RudderCallback = RudderManager(True, lesson_buffer, rudder_lstm,'logs/cl_ppo_rudder/lstm_frames/',env,scaling_factor)
        self.model = MaskablePPO.load('models/cl_ppo_rudder/phase_1a', env=self.env)
        phase_2_callback = CL_PPO_RUDDER_PHASE_2_Callback(True,100,info,'a',99)
        env_logger = EnvLogger(self.args.envlogger_freq, './logs/cl_ppo_rudder/statevar/', self.args.seq_gen, lesson_buffer)

        ## Teach Phase 2a ###
        #Learn 4 party combos
        self.model.learn(total_timesteps=300000, callback=[phase_2_callback,env_logger,auto_save_callback], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_2a')

        ## Teach Phase 2b ###
        #Learn 6 party combos
        phase_2_callback = CL_PPO_RUDDER_PHASE_2_Callback(True,100,info,'b',101)
        self.model.learn(total_timesteps=300000, callback=[phase_2_callback,env_logger,auto_save_callback], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_2b')

        ## Teach Phase 2b ###
        #Learn 8 party combos
        phase_2_callback = CL_PPO_RUDDER_PHASE_2_Callback(True,100,info,'c',64)
        self.model.learn(total_timesteps=300000, callback=[phase_2_callback,env_logger,auto_save_callback], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_2c')

        #Learn uncombine
        phase_2_callback = CL_PPO_RUDDER_PHASE_2_Callback(True,100,info,'d',120)
        self.model.learn(total_timesteps=300000, callback=[phase_2_callback,env_logger,auto_save_callback], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_2d')

        #Learn

    def run_phase_3(self):
        """
        Phase 3 works on learning the wait / deny functions. It does this by starting with an environment where you have
        a butload of walk_ins right at open. The model must learn to prioritize reservations and then give walk_ins
        a wait_time to bring them back


        :return:
        """
        info = {}
        info['tables'] = self.env.tables
        self.mutable_settings['reservations_path'] = 'reservation_files/cl_ppo_rudder/phase_3.csv'
        self.mutable_settings['walk_ins_path'] = 'walk_in_files/cl_ppo_rudder/phase_3.csv'
        self.mutable_settings['end_time'] = 250
        self.mutable_settings['n_steps'] = 1600
        self.mutable_settings['wait_tolerance'] = 10
        self.mutable_settings['phase'] = '3a'

        env = HostWorldEnv(immutable_config=self.immutable_settings, mutable_config=self.mutable_settings)
        env = FlattenObservation(env)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        env = Monitor(env, self.mutable_settings['log_dir'])
        self.env = env
        self.env.reset()

        if self.args.human_player:
            for i in range(1):
                host = BasicHost(env, i)
                data_obs, data_actions = host.run_episode()
                print(len(data_obs))

        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.args.log_dir)
        self.model = MaskablePPO.load('models/cl_ppo_rudder/phase_3a', env=self.env)

        #self.model = MaskablePPO.load('logs/cl_ppo_rudder/best_model.zip', env=self.env)
        phase_3_callback = CL_PPO_RUDDER_PHASE_3_Callback(True,1000,info,'a',120)
        env_logger = EnvLogger(self.args.envlogger_freq, './logs/cl_ppo_rudder/statevar/', self.args.seq_gen, None)

        ## Teach Phase 3a ###
        #Learn all party combos with uncombine and a time setup at the beginning
        self.model.learn(total_timesteps=300000, callback=[phase_3_callback,env_logger,auto_save_callback], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_3a')


    def run_phase_4(self):
        info = {}
        info['tables'] = self.env.tables
        self.mutable_settings['reservations_path'] = 'reservation_files/cl_ppo_rudder/phase_4.csv'
        self.mutable_settings['walk_ins_path'] = 'walk_in_files/cl_ppo_rudder/phase_4.csv'
        self.mutable_settings['end_time'] = 240
        self.mutable_settings['n_steps'] = 1500
        self.mutable_settings['phase'] = '4a'

        env = HostWorldEnv(immutable_config=self.immutable_settings, mutable_config=self.mutable_settings)
        env = FlattenObservation(env)
        env = ActionMasker(env, mask_fn)  # Wrap to enable masking
        env = Monitor(env, self.mutable_settings['log_dir'])
        self.env = env
        self.env.reset()

        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=self.args.log_dir)

        self.model = MaskablePPO.load('models/cl_ppo_rudder/phase_3a', env=self.env)
        #self.model = MaskablePPO.load('logs/cl_ppo_rudder/best_model.zip', env=self.env)
        phase_4_callback = CL_PPO_RUDDER_PHASE_4_Callback(True,1000,info,'a',240)
        env_logger = EnvLogger(self.args.envlogger_freq, './logs/cl_ppo_rudder/statevar/', self.args.seq_gen, None)

        ## Teach Phase 3a ###
        #Learn all party combos with uncombine and a time setup at the beginning
        self.model.learn(total_timesteps=300000, callback=[phase_4_callback,env_logger,auto_save_callback], progress_bar=True)
        self.model.save('models/cl_ppo_rudder/phase_4a')


