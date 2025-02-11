import unittest
from unittest.mock import Mock, patch

import numpy as np
from google import genai

from src.planning.llm_nagivator import GeminiThinkingNavigator, LLMNavigationPlan


class TestGeminiThinkingNavigator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_model = Mock(spec=genai.Client)
        self.mock_chat = Mock()
        self.mock_model.chats.create.return_value = self.mock_chat
        self.navigator = GeminiThinkingNavigator(model=self.mock_model)

    def test_initialization(self):
        """Test proper initialization of GeminiThinkingNavigator."""
        self.assertEqual(self.navigator.model, self.mock_model)
        self.assertEqual(self.navigator.chat, self.mock_chat)
        self.mock_model.chats.create.assert_called_once_with(model="gemini-2.0-flash-thinking-exp")

    def test_plan_successful_execution(self):
        """Test successful execution of plan method with valid input."""
        # Arrange
        test_prompt = "Navigate to point A"
        mock_response = Mock()
        mock_response.text = "[(1.0, 2.0), (3.0, 4.0)]"
        self.mock_chat.send_message.return_value = mock_response

        expected_waypoints = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        # Act
        result = self.navigator.plan(test_prompt)

        # Assert
        self.mock_chat.send_message.assert_called_once_with([test_prompt])
        self.assertIsInstance(result, LLMNavigationPlan)
        np.testing.assert_array_equal(result.waypoints[0], expected_waypoints[0])
        np.testing.assert_array_equal(result.waypoints[1], expected_waypoints[1])
        self.assertEqual(result.prompt, test_prompt)
        self.assertEqual(result.trajectory, [])

    @patch("builtins.eval")
    def test_plan_with_invalid_response(self, mock_eval):
        """Test plan method handling of invalid response format."""
        # Arrange
        mock_eval.side_effect = SyntaxError("Invalid syntax")
        self.mock_chat.send_message.return_value = Mock(text="invalid response")

        # Act & Assert
        with self.assertRaises(SyntaxError):
            self.navigator.plan("test prompt")

    def test_reset_chat(self):
        """Test chat reset functionality."""
        # Act
        self.navigator.reset_chat()

        # Assert
        self.mock_model.chats.create.assert_called()
        self.assertEqual(self.mock_model.chats.create.call_count, 2)  # Initial + reset


class TestLLMNavigationPlan(unittest.TestCase):
    def test_navigation_plan_creation(self):
        """Test creation of LLMNavigationPlan with valid inputs."""
        # Arrange
        waypoints = [np.array([1.0, 2.0])]
        trajectory = []
        prompt = "test prompt"

        # Act
        plan = LLMNavigationPlan(waypoints=waypoints, trajectory=trajectory, prompt=prompt)

        # Assert
        self.assertEqual(plan.waypoints, waypoints)
        self.assertEqual(plan.trajectory, trajectory)
        self.assertEqual(plan.prompt, prompt)

    def test_navigation_plan_immutability(self):
        """Test that LLMNavigationPlan is immutable."""
        # Arrange
        plan = LLMNavigationPlan(waypoints=[np.array([1.0, 2.0])], trajectory=[], prompt="test")

        # Act & Assert
        with self.assertRaises(AttributeError):
            plan.waypoints = []  # Should raise error due to frozen=True


if __name__ == "__main__":
    unittest.main()
