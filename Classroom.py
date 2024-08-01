from HostEnv import HostWorldEnv
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import numpy as np
import pygame
from sb3_contrib import MaskablePPO

class Curriculum(object):
    def __init__(self, lessons, env_config, verbose=0, render=False, render_settings=None):
        """
        :param lessons: list of type lessons
        """
        if render_settings is None and render is True:
            #Default render settings
            render_settings = {
                'window_size': (800, 600),
                'grid_size': 25,
            }

        self.env_config = env_config
        self.lessons = lessons
        self.curr_lesson = 0
        self.report_card = CurriculumStats()


"""
We are using lesson to teach a concept so there may be multiple episodes in each lesson in order 
to get the model to learn the concept. Each lesson ends with a "exam" that the model must pass 
"""
class Lesson(object):
    def __init__(self,episodes,exam,verbose=0,render=False):
        """
        :param episodes: list of type episodes
        :param exam: exam of type episode
        :param exam_max_score: the max possible score of the exam
        :param exam_pass_score: the passable score of the exam
        :param verbose: verbosity level 0-3
        :param render: should we render the lesson?
        """
        self.episodes = episodes
        self.exam = exam
        self.verbose = verbose
        self.render = render

"""
    Episodes are specifc instances of our environment with episode specific rules
    examples include walk_ins, reservations, but also wait_tolerance etc.
    
    Each episode has a max possible score
"""
class Episode(object):

    def __init__(self,episode_settings,max_score):
        """
        :param episode_settings: Episode settings, must be equivalent to env.mutable_config
        :param max_score: the maximum possible score of the episode
        :param pass_score: the passable_score for the episode
        """
        self.episode_settings = episode_settings
        self.max_score = max_score

class Stats(object):

    def add_episode_stats(self,model,episode):
        pass
    pass
class LessonStats(Stats):
    pass
class CurriculumStats(Stats):
    pass

def run_curriculum(model,curriculum,verbose=0,render=False):
    """
    :param model: The model for which to train specified as a path to a model file
    :param curriculum: curriculum object
    :param debug: flag for debugging or not
    :param render: flag for whether to render the model status
    :return: returns report card of how the model did in the curriculum
    """
    return

def run_lesson(model,lesson,verbose=0,render=False,render_settings=None):
    pass

def run_episode(model,episode,env,timesteps,model_stats,verbose=0):

    """
    :param model: PPO Model that we are running the episode with
    :param episode: The episode we want to run the model on
    :param env: The environment we want to run the model on (need to update mutable settings)
    :param timesteps: How many times we want to train the model on the episode
    :param model_stats: Stats object that holds statistics about the model
    :param verbose: verbosity level 0-3
    :return: VOID
    """

    """ 
    Dont create new environment since action and observation space will be the same
    only need to change the mutable parameters with the episode specific settings
    """
    env.set_mutable_config(episode.episode_settings)
    model.set_env(env)

    """
    We want the model to learn until it passes a certain amount of score.
    If model has not surpassed the passing score, we will give up but put a flag for the episode
    """
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False)  # train
    model_stats.add_episode_stats(model,episode)



def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_action_mask()
