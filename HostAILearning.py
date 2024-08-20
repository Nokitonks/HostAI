# Mock imports that are not the focus of unit testing
from HostEnv import HostWorldEnv
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from BasicRestaurant1 import BasicRestaurantTables
from Classroom import Lesson, Episode
from utils.helperFunctions import mask_fn
import pygame
import time
def create_curriculum(env):
    """
    :return: Curriculum object that has lessons made for learning to seat parties at correctly sized tables
    """

    table_size_lesson = Lesson("Table Size Lesson",[],None,30,1500,verbose=1)
    passing_percentage = 90
    episode_settings1 = {
        "clean_time":{2:5,
                      4:5,
                      6:5,
                      8:5},
        "wait_tolerance":10,
        "reservations_path":'reservation_files/reservations(1).csv',
        "log_dir":'.',
        "end_time":50,
        'walk_ins_path':'walk_in_files/walk_ins_none.csv'
    }

    episode_1 = Episode(episode_settings1,19,18,env)
    table_size_lesson.episodes.append(episode_1)
    """
    This episode max score can only be reached after sitting all reservations at appropriate sized tables 
    """
    episode_settings2 = {
        "clean_time":{2:5,
                      4:5,
                      6:5,
                      8:5},
        "wait_tolerance":10,
        "reservations_path":'reservation_files/reservations0.csv',
        "log_dir":'.',
        "end_time":360,
        'walk_ins_path':'walk_in_files/walk_ins_none.csv'
    }
    episode_2 = Episode(episode_settings2,264,260,env)
    table_size_lesson.episodes.append(episode_2)

    """
    This episode incorperates combining and uncombining tables in order to seat the appropriate amount of people
    """

    episode_settings3 = {
        "clean_time":{2:5,
                      4:5,
                      6:5,
                      8:5},
        "wait_tolerance":10,
        "reservations_path":'reservation_files/reservations1.csv',
        "log_dir":'.',
        "end_time":360,
        'walk_ins_path':'walk_in_files/walk_ins_none.csv'
    }
    episode_3 = Episode(episode_settings3,264,260,env)
    table_size_lesson.episodes.append(episode_3)
    table_size_lesson.exam = episode_3
    model = MaskablePPO('MlpPolicy', env, verbose=1, device='mps', tensorboard_log="logs")
    table_size_lesson.run_lesson(model)

def run_random_episode(render=True,sleep=0.0):
    state, info = env.reset()
    done = False
    score = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Use the action mask to filter valid actions
        action_mask = env.get_action_mask()

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
        if render:
            time.sleep(sleep)
            env.render()
    pass
if __name__ == "__main__":


    """
    We begin our trials with a very simple restaurant. 10 tables total. 8 two tops and 2 four tops
    each of the 8 two tops can be combined with four others. and the 2 four tops can also be combined to form an eight top
    """
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
        "log_dir":'',
        "end_time":50,
        'walk_ins_path': 'walk_in_files/walk_ins_none.csv'
    }

    env = HostWorldEnv(immutable_config=immutable_settings,
                       mutable_config=default_mutable_settings)
    env = FlattenObservation(env)
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    #while(True):
    #    run_random_episode()
    curriculum = create_curriculum(env)
    pass