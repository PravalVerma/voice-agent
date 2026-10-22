"""Call state enum, in its own module so session.py and pipeline.py can both
import it without creating a circular import between them."""

import enum


class State(str, enum.Enum):
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    PROCESSING = "processing"
    SPEAKING = "speaking"
