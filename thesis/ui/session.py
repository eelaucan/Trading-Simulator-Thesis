"""Typed UI-side session metadata for local experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum


SUPPORTED_CONDITIONS: tuple[str, ...] = (
    "human_only",
    "human_with_coach_placeholder",
)
CONDITION_LABELS: dict[str, str] = {
    "human_only": "Human Only",
    "human_with_coach_placeholder": "Human + Coach Placeholder",
}


class SessionStatus(str, Enum):
    """High-level lifecycle states for one local human session."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    FINISHED = "finished"


def condition_display_label(condition: str) -> str:
    """Return a participant-friendly label for a stored condition code."""
    return CONDITION_LABELS.get(condition, condition.replace("_", " ").title())


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """Human-session metadata kept outside the simulator core."""

    participant_id: str
    condition: str
    episode_name: str
    dataset_path: str
    started_at: datetime
    decision_start_week: int
    visible_history_weeks_at_start: int
    finished_at: datetime | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        participant_id = self.participant_id.strip()
        condition = self.condition.strip()
        episode_name = self.episode_name.strip()
        dataset_path = self.dataset_path.strip()
        notes = None if self.notes is None else self.notes.strip() or None

        if not participant_id:
            raise ValueError("participant_id must be a non-empty string")
        if not condition:
            raise ValueError("condition must be a non-empty string")
        if not episode_name:
            raise ValueError("episode_name must be a non-empty string")
        if not dataset_path:
            raise ValueError("dataset_path must be a non-empty string")
        if self.decision_start_week < 0:
            raise ValueError("decision_start_week must be >= 0")
        if self.visible_history_weeks_at_start <= 0:
            raise ValueError("visible_history_weeks_at_start must be > 0")

        object.__setattr__(self, "participant_id", participant_id)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "episode_name", episode_name)
        object.__setattr__(self, "dataset_path", dataset_path)
        object.__setattr__(self, "notes", notes)

    def mark_finished(self, finished_at: datetime) -> "SessionMetadata":
        """Return a copy with the session finish timestamp populated."""
        return replace(self, finished_at=finished_at)

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "episode_name": self.episode_name,
            "dataset_path": self.dataset_path,
            "started_at": self.started_at.isoformat(),
            "decision_start_week": self.decision_start_week,
            "visible_history_weeks_at_start": self.visible_history_weeks_at_start,
            "finished_at": self.finished_at.isoformat() if self.finished_at is not None else None,
            "notes": self.notes,
        }
