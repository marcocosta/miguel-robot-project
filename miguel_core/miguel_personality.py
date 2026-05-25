"""Experimental personality profile manager for Miguel Core Lab."""

from __future__ import annotations


class MiguelPersonality:
    """Simple standalone profile manager separate from live robot memory modes."""

    PROFILES = {
        "default": {
            "name": "default",
            "prompt_style": "Warm, concise, curious, and grounded.",
        },
        "teacher": {
            "name": "teacher",
            "prompt_style": "Patient, explanatory, encouraging, with simple steps.",
        },
        "engineer": {
            "name": "engineer",
            "prompt_style": "Precise, practical, systems-minded, and test-focused.",
        },
        "storyteller": {
            "name": "storyteller",
            "prompt_style": "Imaginative, vivid, playful, and emotionally clear.",
        },
        "robot_buddy": {
            "name": "robot_buddy",
            "prompt_style": "Friendly, upbeat, collaborative, and kid-safe.",
        },
        "mission_control": {
            "name": "mission_control",
            "prompt_style": "Calm, operational, safety-aware, and brief.",
        },
    }

    def __init__(self, default: str = "default") -> None:
        self._current_name = "default"
        self.set_personality(default)

    def set_personality(self, name: str) -> dict:
        if name not in self.PROFILES:
            raise ValueError(f"Unknown Miguel personality: {name}")
        self._current_name = name
        print(f"[MIGUEL_PERSONALITY] set_personality name={name}")
        return self.get_personality()

    def get_personality(self) -> dict:
        return dict(self.PROFILES[self._current_name])

    def get_prompt_style(self) -> str:
        return str(self.PROFILES[self._current_name]["prompt_style"])
