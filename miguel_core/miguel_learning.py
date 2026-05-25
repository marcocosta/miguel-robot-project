"""Safe placeholders for future Miguel learning workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class MiguelLearning:
    """Records conservative learning candidates without training any model."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_path = self.data_dir / "miguel_learning_candidates.jsonl"
        self.candidates_path.touch(exist_ok=True)

    def record_interaction(
        self,
        user_text: str,
        assistant_text: str,
        metadata: dict | None = None,
    ) -> dict:
        candidates = self.extract_learning_candidates(user_text, assistant_text)
        record = {
            "timestamp": self._utc_now(),
            "type": "interaction",
            "user_text": user_text,
            "assistant_text": assistant_text,
            "metadata": metadata or {},
            "candidate_count": len(candidates),
            "candidates": candidates,
            "model_training_performed": False,
        }
        self._append_jsonl(record)
        print("[MIGUEL_LEARNING] record_interaction")
        return record

    def extract_learning_candidates(self, user_text: str, assistant_text: str) -> list[dict]:
        candidates: list[dict] = []
        normalized = user_text.strip()
        if normalized:
            label = "user_preference" if any(
                phrase in normalized.lower()
                for phrase in ("remember", "i like", "i prefer", "call me", "my name is")
            ) else "conversation_example"
            candidates.append(self.build_training_example(normalized, assistant_text.strip(), label))
        print(f"[MIGUEL_LEARNING] extract_learning_candidates count={len(candidates)}")
        return candidates

    def build_training_example(
        self,
        user_text: str,
        assistant_text: str,
        label: str | None = None,
    ) -> dict:
        return {
            "timestamp": self._utc_now(),
            "label": label or "unlabeled",
            "input": user_text,
            "output": assistant_text,
            "approved_for_training": False,
            "notes": "Candidate only; no model training happens in Miguel Core Lab.",
        }

    def _append_jsonl(self, record: dict) -> None:
        with self.candidates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
