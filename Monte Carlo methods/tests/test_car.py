import unittest
from unittest.mock import Mock
from car import Car
from environment import Environment

class TestCar(unittest.TestCase):
    def setUp(self):
        self.environment = Environment(filename="tests/fixtures/scenario1.txt")
        self.car = Car(self.environment, select_action_fn=lambda car: (0, 0))

    def test_default_actions(self):
        actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.assertEqual(self.car.actions, actions)

    def test_str(self):
        self.assertEqual(str(self.car), "#")

    def test_reset(self):
        self.car.speed = (2, 2)
        self.position = (1, 5)
        self.environment.select_start_position = Mock(return_value=(0, 0))
        self.car._reset()

        self.assertEqual(self.car.speed, (0, 0))
        self.assertEqual(self.car.position, (0, 0))
        self.environment.select_start_position.assert_called()

    def test_calculate_policy(self):
        Q = {
            "state1": {
                "action1": 1,
                "action2": 2,
                "action3": 0.5,
            },
            "state2": {
            "action1": 0.5
            }
        }

        P = self.car.calculate_policy(Q)
        self.assertEqual(P, {
            'state1': 'action2',
            'state2': 'action1'
        })

    def test_reward_for_finish_cell(self):
        reward = self.car.reward((1, 16))
        self.assertEqual(reward, 0)

    def test_step_normal(self):
        self.car.position = (7, 5)
        self.car.speed = (0, 0)
        self.car.select_action = Mock(return_value=(1, 0))
        reward, action = self.car.step()
        self.assertEqual((-1, (1, 0)), (reward, action))
        self.assertEqual(self.car.position, (8, 5))
        self.assertEqual(self.car.speed, (1, 0))

    def test_step_speeds_over_maximum(self):
        self.car.position = (3, 5)
        self.car.speed = (4, 1)
        self.car.select_action = Mock(return_value=(1, 0))
        reward, action = self.car.step()
        self.assertEqual((-1, (1, 0)), (reward, action))
        self.assertEqual(self.car.position, (7, 6))
        self.assertEqual(self.car.speed, (4, 1))

    def test_step_speeds_under_minimum(self):
        self.car.position = (3, 10)
        self.car.speed = (0, -4)
        self.car.select_action = Mock(return_value=(0, -1))
        reward, action = self.car.step()
        self.assertEqual((-1, (0, -1)), (reward, action))
        self.assertEqual(self.car.position, (3, 6))
        self.assertEqual(self.car.speed, (0, -4))

    def test_step_out_of_board_above(self):
        self.car.position = (3, 10)
        self.car.speed = (-3, -4)
        self.car.select_action = Mock(return_value=(-1, 0))
        self.environment.select_start_position = Mock(return_value=(9, 7))
        reward, action = self.car.step()
        self.assertEqual((-1, (-1, 0)), (reward, action))
        self.assertEqual(self.car.position, (9, 7))
        self.assertEqual(self.car.speed, (0, 0))

    def test_step_out_of_board_below(self):
        self.car.position = (7, 7)
        self.car.speed = (5, 0)
        self.car.select_action = Mock(return_value=(-1, 0))
        self.environment.select_start_position = Mock(return_value=(9, 8))
        reward, action = self.car.step()
        self.assertEqual((-1, (-1, 0)), (reward, action))
        self.assertEqual(self.car.position, (9, 8))
        self.assertEqual(self.car.speed, (0, 0))

    def test_step_out_of_board_left(self):
        self.car.position = (4, 1)
        self.car.speed = (0, -1)
        self.car.select_action = Mock(return_value=(0, -1))
        self.environment.select_start_position = Mock(return_value=(9, 8))
        reward, action = self.car.step()
        self.assertEqual((-1, (0, -1)), (reward, action))
        self.assertEqual(self.car.position, (9, 8))
        self.assertEqual(self.car.speed, (0, 0))

    def test_step_out_of_board_right(self):
        self.car.position = (5, 16)
        self.car.speed = (-1, 0)
        self.car.select_action = Mock(return_value=(0, 1))
        self.environment.select_start_position = Mock(return_value=(9, 8))
        reward, action = self.car.step()
        self.assertEqual((-1, (0, 1)), (reward, action))
        self.assertEqual(self.car.position, (9, 8))
        self.assertEqual(self.car.speed, (0, 0))

    def test_step_finish(self):
        self.car.position = (1, 15)
        self.car.speed = (-1, 0)
        self.car.select_action = Mock(return_value=(0, 1))
        reward, action = self.car.step()
        self.assertEqual((0, (0, 1)), (reward, action))
        self.assertEqual(self.car.position, (0, 16))
        self.assertEqual(self.car.speed, (-1, 1))

    def test_step_cross_finish(self):
        self.car.position = (2, 13)
        self.car.speed = (0, 3)
        self.car.select_action = Mock(return_value=(0, 1))
        reward, action = self.car.step()
        self.assertEqual((0, (0, 1)), (reward, action))
        self.assertEqual(self.car.position, (2, 16))
        self.assertEqual(self.car.speed, (0, 4))

    def test_reward_for_non_finish_cell(self):
        self.assertEqual(self.car.reward((1, 15)), -1)
        self.assertEqual(self.car.reward((0, 0)), -1)
        self.assertEqual(self.car.reward((9, 5)), -1)

    def test_play(self):
        self.car.select_action = Mock(return_value=(0, 1))
        self.environment.select_start_position = Mock(return_value=(1, 3))
        steps, rewards = self.car.play()

        self.assertEqual(steps, [
            (((1, 4), (0, 1)), (0, 1)),
            (((1, 6), (0, 2)), (0, 1)),
            (((1, 9), (0, 3)), (0, 1)),
            (((1, 13), (0, 4)), (0, 1)),
            (((1, 16), (0, 4)), (0, 1))
        ])

        self.assertEqual(rewards, [-1, -1, -1, -1, 0])

    def test_train(self):
        self.car.select_action = Mock(return_value=(0, 1))
        self.environment.select_start_position = Mock(return_value=(1, 3))
        self.car.train(1)

        self.assertEqual(self.car.position, (1, 16))
        self.assertEqual(self.car.speed, (0, 4))
        
        self.assertEqual(self.car.Q[((1, 4), (0, 1))], {(0, 1): -4})
        self.assertEqual(self.car.returns_sum[((1, 4), (0, 1))], -4)
        self.assertEqual(self.car.returns_count[((1, 4), (0, 1))], 1)

    def test_calculate_returns_only_first(self):
        steps = [
            (((2, 3), (-3, 1)), (1, 0)),
            (((0, 3), (-3, 1)), (0, 1)),
            (((2, 3), (-3, 1)), (-1, 1)),
            (((1, 2), (2, 1)), (1, 0)),
            (((2, 3), (-3, 1)), (1, 0)),
        ]

        rewards = [
            -1, -1, -1, -1, -1
        ]

        R = self.car._calculate_returns(steps, rewards)
        self.assertEqual(list(R.keys()), [(((2, 3), (-3, 1)), (1, 0)),
            (((1, 2), (2, 1)), (1, 0)),
            (((2, 3), (-3, 1)), (-1, 1)),
            (((0, 3), (-3, 1)), (0, 1))
        ])

        self.assertEqual(R[(((2, 3), (-3, 1)), (1, 0))], -5)

    def test_select_action(self):
        car = Car(self.environment, select_action_fn=lambda car: (-501, -660))
        self.assertEqual(car.select_action(), (-501, -660))

