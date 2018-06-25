import unittest
from unittest.mock import patch
import sys
from src.utility import *
import os


class TestUtility(unittest.TestCase):

    def test_input_read_correct(self):
        user_input = ["prog ", 'test_arms.txt', 'test_content.txt']

        expected_dict = {
            109568 : [
                1.0, 0.8750070869361845, 0.8625865257428182,
                0.5165442987577996, 0.8545485350877207, 0.7084979088758754
            ]
        }

        with patch.object(sys, 'argv', user_input):
            arms = read_input_arms()
            print(arms)
        self.assertEqual(arms, expected_dict)



if __name__ == '__main__':
    unittest.main()

