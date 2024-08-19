from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from BasicRestaurant1 import BasicRestaurantTables
from HostEnv import HostWorldEnv
import wandb
from wandb.integration.sb3 import WandbCallback

def train(seed, args, shared_list):


    run_name = f"Basic_Host_env"
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
            env = Monitor(env, default_mutable_settings['log_dir'] + f"{rank}")
            return env
        return _init

    num_env = 3
    env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    config = {
        "policy_type": "MlpPolicy",
        'total_timesteps': args.total_timesteps,
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


