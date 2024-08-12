import numpy as np

# Mock imports that are not the focus of unit testing
from HostEnv import HostWorldEnv
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from Classroom import Curriculum, Lesson, Episode, TrialStats, LessonStats, EpisodeStats, CurriculumStats

def create_curriculum():
    """
    :return: Curriculum object that has lessons made for learning to seat parties at correctly sized tables
    """

    table_size_lesson = Lesson("Table Size Lesson",[],None,10,1000)
    passing_percentage = 90
    default_mutable_settings = {
        "max_time":360,
        "clean_time":8,
        "wait_tolerance":10,
        "window_size":650,
        "grid_size":10
    }
    episode_1 = Episode()


if __name__ == "__main__":

    curriculum = create_curriculum()
    pass