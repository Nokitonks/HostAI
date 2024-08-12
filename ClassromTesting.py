import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Mock imports that are not the focus of unit testing
from HostEnv import HostWorldEnv
from gymnasium.wrappers import FlattenObservation
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from Classroom import Curriculum, Lesson, Episode, TrialStats, LessonStats, EpisodeStats, CurriculumStats


class TestCurriculum(unittest.TestCase):

    @patch('Classroom.evaluate_policy')
    def test_classroom(self,mock_evaluate_policy):

        """
        Starting from the bottom of the heirchy in Stats testing
        :return:
        """

        """
        STATS TESTING
        """
        trial_stat1 = TrialStats(5,1,80,True)
        with self.assertRaises(AssertionError) as context:
            trial_stat2 = TrialStats(5, 1, 180,True)

        """
        Episode phase here... 
        """
        model = Mock(spec=MaskablePPO)
        ep_settings = Mock()


        episode1 = Episode(ep_settings,10,7,Mock(HostWorldEnv),10)
        #Set the model to past tests on first try

        mock_evaluate_policy.return_value = [8,0]
        episode1.run_trial(model,10)
        self.assertTrue(episode1.passed)
        self.assertEqual(len(episode1.stats.trial_stats),1)

        #Change episode to fail
        mock_evaluate_policy.return_value = [6,0]
        episode1.run_trial(model,10)
        self.assertFalse(episode1.passed)
        self.assertEqual(len(episode1.stats.trial_stats),2)

        episode2 = Episode(ep_settings,10,8,Mock(HostWorldEnv),10)

        """
        Lesson phase here...
        """
        mock_evaluate_policy.return_value = [7,0]
        #Should pass our lesson but fail the exam
        lesson1 = Lesson("lesson1",[episode1],episode2,5,100)
        lesson_states = lesson1.run_lesson(model)
        self.assertTrue(episode1.passed)
        self.assertFalse(lesson1.exam.passed)
        self.assertFalse(lesson_states.exam_stats.passed)

        #Should not be able to pass the lesson
        mock_evaluate_policy.return_value = [2,0]
        lesson1 = Lesson("lesson1",[episode1],episode2,5,100)
        lesson_states = lesson1.run_lesson(model)
        self.assertFalse(episode1.passed)
        self.assertFalse(lesson1.exam.passed)




if __name__ == "__main__":
    unittest.main()