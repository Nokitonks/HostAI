from faker.providers.address import th
import torch
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from imitation.data import types, rollout
from imitation.algorithms import bc
from BasicHost import BasicHost
from BasicRestaurant1 import BasicRestaurantTables, MBPost
from HostEnv import HostWorldEnv, mask_fn
import wandb
from wandb.integration.sb3 import WandbCallback
import datetime
from rudder import LessonBuffer, RRLSTM
from utils.callbacks import EnvLogger, SaveOnBestTrainingRewardCallback, ProgressBarManager, RudderManager
from csv_generators import *


def train(seed, args, shared_list):


    run_name = f"Basic_Host_env"
    log_dir_statevar = args.log_dir + 'statevar/'

    immutable_settings = {
        'tables': MBPost().tables,
        'max_party_size': 8,
        'max_time': 100,
        'max_wait_list': 80,
        'max_res_list': 80,
        'window_size': (1240, 1080),
        "grid_size": 35,
        'n_steps': args.env_steps,
        'wait_quote_min':10,
        'wait_quote_max': 90,
        'wait_quote_step': 5
    }
    default_mutable_settings = {
        "clean_time": {2: 1,
                       4: 5,
                       6: 5,
                       8: 5,
                       10: 5,
                       12: 5,
                       14: 5,
                       16: 5,
                       18: 5,
                       20: 5,
                       22: 5,
                       24: 5,
                       },
        "wait_tolerance": 10,
        "reservations_path": 'reservation_files/reservations(5).csv',
        "log_dir": args.log_dir,
        "end_time": 80,
        'walk_ins_path': 'walk_in_files/walk_ins.csv',
        'num_servers':2,
        'server_sections':{'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0,'10':0,'11':0,'12':0,
                           '20': 0, '21': 0, '22': 0, '23': 0, '24': 0, '25': 0, '26': 0,
                           '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0,
                           '40': 0, '41': 0, '42': 0, '43': 0,
                           '51': 1, '52': 1, '53': 1, '54': 1, '55': 1, '56': 1, '61': 1,'62':1,
                           '71': 1, '72': 1, '73': 1, '74': 1, '75': 1, '76': 1,
                           '86': 1, '81': 1, '82': 1, '83': 1, '84': 1, '85': 1,
                           '90': 1, '91': 1, '92': 1, '93': 1, '94': 1, '95': 1,
                           '96': 1
                           },
        'CL_step': args.CL_step
    }
    env = HostWorldEnv(immutable_config=immutable_settings, mutable_config=default_mutable_settings)
    env = FlattenObservation(env)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env = Monitor(env, default_mutable_settings['log_dir'])
    print("Observation shape:", env.observation_space.shape)

    num_its = 2
    bc_data_obs = []
    bc_data_actions = []
    infos = []
    dones = []
    if args.human_player:
        for i in range(num_its):
            host = BasicHost(env,i)
            data_obs, data_actions = host.run_episode()
            print(len(data_obs))
            bc_data_obs.append(data_obs)
            bc_data_actions.append(data_actions)


    """
    Function to create environment and wrap it with a monitor
    """
    def make_env(rank):

        def _init():
            immutable_settings = {
                'tables': BasicRestaurantTables().tables,
                'max_party_size': 8,
                'max_time': 500,
                'max_wait_list':10,
                'max_res_list':15,
                'window_size':(640,480),
                "grid_size": 50
            }
            default_mutable_settings = {
                "clean_time":{2:1,
                              4:10,
                              6:20,
                              8:20},
                "wait_tolerance":1,
                "reservations_path": 'reservation_files/reservations(1).csv',
                "log_dir":args.log_dir,
                "end_time":50,
                'walk_ins_path': 'walk_in_files/walk_ins_none.csv',
                'CL_step':args.CL_step
            }
            env = HostWorldEnv(immutable_config=immutable_settings, mutable_config=default_mutable_settings)
            env = FlattenObservation(env)
            env = ActionMasker(env, mask_fn)  # Wrap to enable masking
            env = Monitor(env, default_mutable_settings['log_dir'] + f"{rank}")
            return env
        return _init

    num_env = 3
    #env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    config = {
        "policy_type": "MlpPolicy",
        'total_timesteps': args.total_timesteps,

        "policy_kwargs": dict(
            activation_fn=torch.nn.Sigmoid),

        # "env_name": "CartPole-v1",
        "batch_size": args.batch_size,
    }
    if args.track_wandb:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            # config=vars(args),
            config=config,
            name=run_name,
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

    if args.track_wandb:
        tensorboard_log = f"runs/{run.id}"
    else:
        tensorboard_log = args.log_dir + "/tensorboard/"

    print("args.CL_step", args.CL_step)

    """
    Start with BC to initalize our model
    """
    if args.bc:
        # Flatten the (10, 25, 40165) observations to (250, 40165)
        reshaped_observations = np.array(bc_data_obs).reshape(num_its * args.env_steps, env.observation_space.shape[0])

        # Similarly, flatten the actions from (10, 25) to (250,)
        reshaped_actions = np.array(bc_data_actions).reshape(num_its * args.env_steps)
        infos = [{} for _ in range(len(reshaped_observations))]
        dones = np.zeros(len(reshaped_observations), dtype=bool)
        # Assuming you already have your human demonstrations in the form of `observations` and `actions`
        # Create a demonstration dataset
        transitions = types.Transitions(
            obs=reshaped_observations,
            acts=reshaped_actions,
            next_obs=np.zeros_like(reshaped_observations),  # Dummy next observations (not needed)
            dones=dones,
            infos=np.array(infos)
        )



    model = MaskablePPO("MlpPolicy",
                env,
                seed=seed,
                verbose=1,
                ent_coef=args.ent_coef,
                gamma=args.gamma,
                n_steps=args.n_steps,
                tensorboard_log=tensorboard_log,
                target_kl=args.target_kl,
                policy_kwargs=config["policy_kwargs"],
                n_epochs=args.n_epochs,
                learning_rate=args.learning_rate,
                max_grad_norm=0.5,
                )
    #model = MaskablePPO.load(args.log_dir + "ShortEpModel.zip", env=env)

    if args.bc:

        # Initialize behavior cloning with the same policy as PPO
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            policy=model.policy,
            rng=np.random.default_rng(0),
            batch_size=2
        )

        # Pretrain the PPO model using behavior cloning from human demonstrations
        bc_trainer.train(n_epochs=100)  # Adjust the number of epochs as needed

    print("target kl and clip range are set to:", args.target_kl, args.clip_range)

    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=args.log_dir)


    #Lesson buffer for RUDDER learning
    lb_size = 264
    n_lstm = 16
    max_time = 100
    policy_lr = 0.1
    lstm_lr = 1e-2
    l2_regularization = 1e-6
    avg_window = 750

    print(f"Action size = {env.get_n_actions()[-1]}, State size = {env.get_state_shape()[-1]}\n")
    lesson_buffer = LessonBuffer(size=lb_size, max_time=max_time, n_features=env.get_state_shape()[-1])

    rudder_lstm = RRLSTM(state_input_size=env.get_state_shape()[-1], n_actions=env.get_n_actions()[-1], buffer=lesson_buffer, n_units=n_lstm,
                       lstm_lr=lstm_lr, l2_regularization=l2_regularization, return_scaling=10,
                       lstm_batch_size=8, continuous_pred_factor=0.5)
    RudderCallback = RudderManager(False,lesson_buffer,rudder_lstm)
    start = datetime.datetime.now()

    with ProgressBarManager(config["total_timesteps"]) as progress_callback:
        if args.track_wandb and args.envlogger:
            callbacks = [progress_callback, auto_save_callback, EnvLogger(args.envlogger_freq, log_dir_statevar,args.seq_gen,lesson_buffer)]
        elif args.track_wandb:
            callbacks = [progress_callback, wandbc, auto_save_callback]

        elif args.track_local and args.envlogger:
            callbacks = [progress_callback, auto_save_callback, EnvLogger(args.envlogger_freq, log_dir_statevar,args.seq_gen,lesson_buffer)]
        elif args.track_local:
            callbacks = [progress_callback, auto_save_callback]

        # print callbacks used
        print("Callbacks used are:", callbacks)
        model.learn(total_timesteps=config["total_timesteps"], callback=callbacks, progress_bar=True)

    print("Total training time: ", datetime.datetime.now() - start)

    """
    # del the latest model and load the model with best episodic reward
    del model
    model = MaskablePPO.load(args.log_dir + "basic_model.zip", env=env)

    if args.track_wandb:
        model.save(f"models/{run.id}")  # save the best model to models folder locally
        print(f'saved the best model to wandb')
    print("loaded the model with best reward")
    # plot training volume and action distributions
    if args.verbose_logging:
        if args.track_wandb:
            plt = plot_volume_and_action_distributions(monitor_dir=args.log_dir + 'monitor.csv',
                                                       fig_name='train_vol_action_distrib_' + run_name,
                                                       save_fig=False, plot_wandb=args.track_wandb, run=run)

        else:
            run = 0
            plt = plot_volume_and_action_distributions(monitor_dir=args.log_dir + 'monitor.csv',
                                                       fig_name='train_vol_action_distrib_' + run_name,
                                                       save_fig=False, plot_wandb=args.track_wandb, run=run)
            plt.show()
    else:
        print("No episode plotting.")

    verbose_logging = args.verbose_logging
    env = SutcoEnv(600,
                   verbose_logging,
                   args.bunkers,
                   args.use_min_rew,
                   args.number_of_presses,
                   args.CL_step)

    if verbose_logging:
        env = Monitor(env, args.log_dir, info_keywords=("action", "volumes"))
    else:
        env = Monitor(env, args.log_dir)

    # do inference with the best trained agent and plot state variables
    volumes, actions, rewards, t_p1, t_p2 = inference(args=args,
                                                      log_dir=log_dir,
                                                      deterministic_policy=args.inf_deterministic,
                                                      max_episode_length=600,
                                                      env=env,
                                                      shared_list=shared_list,
                                                      seed=seed)

    # plot the state variables from inference

    if args.track_wandb:
        plot_wandb(env,
                   volumes,
                   actions,
                   rewards,
                   bunker_names
                   )

        plot_episodic_obs(log_dir_statevar, n_bunkers=env.unwrapped.n_bunkers)

    results_path = log_dir
    if args.track_local:
        plot_local_inference(env=env,
                             volumes=volumes,
                             actions=actions,
                             rewards=rewards,
                             seed=seed,
                             fig_name=run_name,
                             save_fig=args.save_inf_fig,
                             color="blue",
                             fig_dir=log_dir,
                             upload_inf_wandb=args.track_wandb,
                             t_p1=t_p1,
                             t_p2=t_p2,
                             results_path=results_path)

        plot_local_voldiff(env=env,
                           volumes=volumes,
                           actions=actions,
                           rewards=rewards,
                           seed=seed,
                           fig_name=run_name,
                           save_fig=args.save_inf_fig,
                           color="blue",
                           bunker_names=bunker_names,
                           fig_dir=log_dir,
                           upload_inf_wandb=args.track_wandb,
                           shared_list=shared_list,
                           args=args)

    """
    run.finish()

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
        n_steps = 64
        target_kl = None
        n_epochs = 1
        learning_rate = 0.0011
        envlogger = True
        envlogger_freq = 100
        clip_range = 0.2
        track_local = True
        batch_size = 64
        env_steps = 100
        seq_gen = True
        human_player = False
        bc = False


    create_reservation_list(5,100,65,1,[0])
    train(0,args,[])