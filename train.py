from faker.providers.address import th
import torch
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.helperFunctions import mask_fn
from BasicRestaurant1 import BasicRestaurantTables
from HostEnv import HostWorldEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import datetime

from utils.callbacks import EnvLogger, SaveOnBestTrainingRewardCallback, ProgressBarManager


def train(seed, args, shared_list):


    run_name = f"Basic_Host_env"
    log_dir_statevar = args.log_dir + 'statevar/'

    immutable_settings = {
        'tables': BasicRestaurantTables().tables,
        'max_party_size': 8,
        'max_time': 500,
        'max_wait_list': 50,
        'max_res_list': 50,
        'window_size': (640, 480),
        "grid_size": 50
    }
    default_mutable_settings = {
        "clean_time": {2: 1,
                       4: 10,
                       6: 20,
                       8: 20},
        "wait_tolerance": 10,
        "reservations_path": 'reservation_files/reservations(1).csv',
        "log_dir": args.log_dir,
        "end_time": 30,
        'walk_ins_path': 'walk_in_files/walk_ins_none.csv',
        'CL_step': args.CL_step
    }
    env = HostWorldEnv(immutable_config=immutable_settings, mutable_config=default_mutable_settings)
    env = FlattenObservation(env)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env = Monitor(env, default_mutable_settings['log_dir'])
    """
    Function to create environment and wrap it with a monitor
    """
    def make_env(rank):

        def _init():
            immutable_settings = {
                'tables': BasicRestaurantTables().tables,
                'max_party_size': 8,
                'max_time': 500,
                'max_wait_list':50,
                'max_res_list':50,
                'window_size':(640,480),
                "grid_size": 50
            }
            default_mutable_settings = {
                "clean_time":{2:1,
                              4:10,
                              6:20,
                              8:20},
                "wait_tolerance":10,
                "reservations_path": 'reservation_files/reservations(1).csv',
                "log_dir":args.log_dir,
                "end_time":50,
                'walk_ins_path': 'walk_in_files/walk_ins_none.csv',
                'CL_step':args.CL_step
            }
            env = HostWorldEnv(immutable_config=immutable_settings, mutable_config=default_mutable_settings)
            print("Here")
            env = Monitor(env, default_mutable_settings['log_dir'] + f"{rank}")
            return env
        return _init

    num_env = 3
    # env = SubprocVecEnv([make_env(i) for i in range(num_env)])
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
                )

    print("target kl and clip range are set to:", args.target_kl, args.clip_range)

    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=args.log_dir)


    start = datetime.datetime.now()

    with ProgressBarManager(config["total_timesteps"]) as progress_callback:
        if args.track_wandb and args.envlogger:
            callbacks = [progress_callback, auto_save_callback, EnvLogger(args.envlogger_freq, log_dir_statevar)]
        elif args.track_wandb:
            callbacks = [progress_callback, wandbc, auto_save_callback]

        elif args.track_local and args.envlogger:
            callbacks = [progress_callback, auto_save_callback, EnvLogger(args.envlogger_freq, log_dir_statevar)]
        elif args.track_local:
            callbacks = [progress_callback, auto_save_callback]
        # model.learn(total_timesteps=config["total_timesteps"], callback=[progress_callback,wanbc,eval_callback])

        # print callbacks used
        print("Callbacks used are:", callbacks)
        model.learn(total_timesteps=config["total_timesteps"], callback=EnvLogger(args.envlogger_freq,log_dir_statevar), progress_bar=True)

    print("Total training time: ", datetime.datetime.now() - start)

    """
    # del the latest model and load the model with best episodic reward
    del model
    model = MaskablePPO.load(args.log_dir + "best_model.zip", env=env)

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
        learning_rate = 0.001
        envlogger = True
        envlogger_freq = 100
        clip_range = 0.2
        track_local = True
        batch_size = 64

    train(0,args,[])