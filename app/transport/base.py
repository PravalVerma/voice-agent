"""Audio transport abstraction.

The pipeline only ever talks to an AudioTransport. The browser WebSocket
implementation is the first one; a telephony adapter (Twilio/Exotel/SIP)
implements the same interface later — including any codec/sample-rate
conversion — without touching the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AudioFrame:
    """Inbound caller audio: raw PCM16 mono at in_sample_rate."""

    pcm: bytes


@dataclass
class ControlMessage:
    """Inbound control message (already parsed)."""

    data: dict


@dataclass
class Disconnected:
    """The caller went away."""


TransportEvent = AudioFrame | ControlMessage | Disconnected


class AudioTransport(ABC):
    #: rate of PCM produced by recv()
    in_sample_rate: int = 16000
    #: rate of PCM expected by send_audio()
    out_sample_rate: int = 24000

    @abstractmethod
    async def recv(self) -> TransportEvent:
        """Wait for the next event from the caller."""

    @abstractmethod
    async def send_audio(self, turn_id: int, pcm: bytes) -> None:
        """Send agent audio (raw PCM16 mono at out_sample_rate) tagged with its turn."""

    @abstractmethod
    async def send_control(self, data: dict) -> None:
        """Send a control message to the caller."""

    @abstractmethod
    async def close(self) -> None: ...
