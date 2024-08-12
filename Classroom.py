from HostEnv import HostWorldEnv
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import numpy as np
import pygame
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

class Curriculum(object):
    def __init__(self, lessons, model, model_save_dir, verbose=0, render=False, render_settings=None):
        """
        :param lessons: list of type lessons
        """
        if render_settings is None and render is True:
            #Default render settings
            render_settings = {
                'window_size': (800, 600),
                'grid_size': 25,
            }
        self.lessons = lessons
        self.verbose = verbose
        self.report_card = CurriculumStats()
        self.model = model
        self.model_save_dir = model_save_dir

    def run_curriculum(self):
        """
        :param model: The model for which to train specified as a path to a model file
        :param curriculum: curriculum object
        :param debug: flag for debugging or not
        :param render: flag for whether to render the model status
        :return: returns report card of how the model did in the curriculum
        """
        for index, lesson in enumerate(self.lessons):

            self.report_card.lesson_stats.append(lesson.run_lesson())

            #Save version of model after every lesson
            self.model.save(f"{self.model_save_dir}/{lesson.name}:Lesson_Number_{index + 1}")


"""
We are using lesson to teach a concept so there may be multiple episodes in each lesson in order 
to get the model to learn the concept. Each lesson ends with a "exam" that the model must pass 
"""
class Lesson(object):
    def __init__(self,name,episodes,exam,max_ep_trials,trial_length,verbose=0,render=False):
        """
        :param episodes: list of type episodes
        :param exam: exam of type episode
        :param exam_max_score: the max possible score of the exam
        :param exam_pass_score: the passable score of the exam
        :param verbose: verbosity level 0-3
        :param render: should we render the lesson?
        """
        self.name = name
        self.episodes = episodes
        self.exam = exam
        self.verbose = verbose
        self.render = render
        self.max_ep_trials = max_ep_trials
        self.trial_length = trial_length
        self.lesson_stats = LessonStats(episodes_stats=[])

    def run_lesson(self,model):

        # Loop through each episode in the lesson
        for episode in self.episodes:

            #Get the stats for the episode
            episode_tries = 0
            while episode_tries < self.max_ep_trials:

                episode.run_trial(model,timesteps=self.trial_length,verbose=self.verbose)
                if not episode.passed:
                    episode_tries += 1
                    continue

                #Append stats to our lesson stats
                self.lesson_stats.episodes_stats.append(episode.stats)
                break

        #Time for the final EXAM, only get to try this one once
        self.exam.add_trial_stats(model)

        self.lesson_stats.exam_stats = self.exam.stats.trial_stats[0]
        return self.lesson_stats


"""
    Episodes are specifc instances of our environment with episode specific rules
    examples include walk_ins, reservations, but also wait_tolerance etc.
    
    Each episode has a max possible score
"""
class Episode(object):

    def __init__(self,episode_settings,max_score,passing_score,env,eval_iters=10):
        """
        :param episode_settings: Episode settings, must be equivalent to env.mutable_config
        :param max_score: the maximum possible score of the episode
        :param pass_score: the passable_score for the episode
        """
        self.episode_settings = episode_settings
        self.max_score = max_score
        self.env = env
        self.stats = EpisodeStats()
        self.eval_iters = eval_iters
        self.passing_score = passing_score
        self.passed = False


    def run_trial(self,model,timesteps,verbose=0):

        """
        :param model: PPO Model that we are running the episode with
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
        self.env.set_mutable_config(self.episode_settings)
        model.set_env(self.env)

        """
        We want the model to learn until it passes a certain amount of score.
        If model has not surpassed the passing score, we will give up but put a flag for the episode
        """
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)  # train
        self.add_trial_stats(model)

    def add_trial_stats(self,model):
        mean, std = evaluate_policy(model,self.env,self.eval_iters)

        #Add other statistics
        percentage_of_max_score = mean / self.max_score * 100

        if mean >= self.passing_score:
            self.passed = True
        else:
            self.passed = False
        trail_stats = TrialStats(mean,std,percentage_of_max_score,self.passed)
        self.stats.trial_stats.append(trail_stats)


class TrialStats(object):

    def __init__(self,mean_reward,std_reward,percentage_of_max_score,passed):
        self.mean_reward = mean_reward
        self.mean_std = std_reward
        self.percentage_of_max_score = percentage_of_max_score
        self.passed = passed
        assert percentage_of_max_score > 0 and percentage_of_max_score <= 100

class LessonStats(object):

    def __init__(self,episodes_stats):
        """
        :param episodes_stats: List containing episode stats objects
        """
        self.episodes_stats = episodes_stats
        self.exam_stats = None
    pass
class EpisodeStats(object):

    def __init__(self):
        self.trial_stats = []


class CurriculumStats(object):

    def __init__(self):
        self.lesson_stats = []






def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_action_mask()
