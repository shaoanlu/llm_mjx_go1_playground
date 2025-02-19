import unittest

from src.control.models.simple_robot import Simple2DRobot, Simple2DRobotConfig


class TestSimple2DRobot(unittest.TestCase):
    def setUp(self):
        pass

    def test_instantiation(self):
        # Test direct creation
        config = Simple2DRobotConfig(a=1, b=2, x_dim=3, u_dim=4)
        robot = Simple2DRobot(config=config)
        self.assertIsInstance(robot, Simple2DRobot)


if __name__ == "__main__":
    unittest.main()
