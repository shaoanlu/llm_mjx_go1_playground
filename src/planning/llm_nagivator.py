from typing import List, Tuple

import numpy as np
from google import genai

from src.planning.base import Planner


class GeminiThinkingNavigator(Planner):
    def __init__(
        self,
        model: genai.Client,
        model_name: str = "gemini-2.0-flash-thinking-exp",
    ):
        self.model = model
        self.chat = self.model.chats.create(model=model_name)

    def plan(self, prompt: str, **kwargs) -> List[np.ndarray]:
        response = self.chat.send_message([prompt])
        waypoints = eval(_clean_instruction(response.text))  # Convert string to list
        return _convert_to_list_of_numpy_arrays(waypoints)

    def reset_chat(self):
        self.chat = self.model.chats.create(model=self.model_name)


def _clean_instruction(instr: str) -> str:
    char_to_be_deleted = ["\n", "\t", "'", "`", "{", "}"]
    for c in char_to_be_deleted:
        instr = instr.replace(c, "")
    return instr


def _convert_to_list_of_numpy_arrays(waypoints: List[Tuple]) -> List[np.ndarray]:
    return [np.array(wp) for wp in waypoints]
