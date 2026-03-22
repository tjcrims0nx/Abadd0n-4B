"""
Media pipeline: images/audio/video, transcription hooks, size caps, temp file lifecycle.
"""

import tempfile
from pathlib import Path
from typing import BinaryIO

# Size caps (bytes)
MEDIA_MAX_IMAGE = 10 * 1024 * 1024   # 10 MB
MEDIA_MAX_AUDIO = 50 * 1024 * 1024   # 50 MB
MEDIA_MAX_VIDEO = 100 * 1024 * 1024  # 100 MB


def get_temp_dir() -> Path:
    """Return temp directory for media lifecycle."""
    return Path(tempfile.gettempdir()) / "abaddon_media"
