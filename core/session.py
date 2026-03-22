"""
Session model: main for direct chats, group isolation, activation modes, queue modes, reply-back.
Group rules: Groups.
"""

from enum import Enum
from typing import TypedDict


class SessionMode(str, Enum):
    """Session activation mode."""
    MAIN = "main"       # Direct chats
    GROUP = "group"     # Group isolation
    QUEUE = "queue"     # Queue modes
    REPLY_BACK = "reply_back"


class SessionConfig(TypedDict, total=False):
    """Session configuration."""
    mode: SessionMode
    group_id: str | None
    activation: str
    queue_enabled: bool
