"""
Abadd0n core platform.

Gateway WS control plane: sessions, presence, config, cron, webhooks, Control UI, Canvas host.
CLI: gateway, agent, send, onboarding, doctor.
Agent runtime: RPC mode with tool streaming and block streaming.
Session model: main (direct chats), group isolation, activation modes, queue modes, reply-back.
Media pipeline: images/audio/video, transcription hooks, size caps, temp file lifecycle.
"""

__all__ = [
    "run_gateway",
    "run_agent",
    "run_send",
    "run_onboarding",
    "run_doctor",
]

from core.gateway import run_gateway
from core.agent import run_agent
from core.send import run_send
from core.onboarding import run_onboarding
from core.doctor import run_doctor
