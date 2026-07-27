"""Teacher scoring: turning student states into token-level supervision."""

from __future__ import annotations

from miniverl.teachers.base import TeacherScorer, TeacherScoreResult
from miniverl.teachers.local import LocalTeacherScorer

__all__ = ["TeacherScorer", "TeacherScoreResult", "LocalTeacherScorer"]
