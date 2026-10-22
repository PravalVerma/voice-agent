"""Browser WebSocket transport.

Wire format:
- client -> server binary: headerless PCM16 mono @ 16 kHz (512-sample frames)
- server -> client binary: 4-byte LE uint32 turn_id + PCM16 mono @ out_sample_rate
- both directions text: JSON control messages
"""

import json
import struct

from fastapi import WebSocket, WebSocketDisconnect

from .base import AudioFrame, AudioTransport, ControlMessage, Disconnected, TransportEvent


class BrowserWebSocketTransport(AudioTransport):
    def __init__(self, ws: WebSocket, out_sample_rate: int = 24000):
        self._ws = ws
        self.in_sample_rate = 16000
        self.out_sample_rate = out_sample_rate
        # Set by main.ws_call when call recording is enabled. Tapping here
        # rather than in the pipeline means both the standard and Gemini Live
        # paths are recorded without either of them knowing about it.
        self.recorder = None

    async def recv(self) -> TransportEvent:
        try:
            message = await self._ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            return Disconnected()
        if message["type"] == "websocket.disconnect":
            return Disconnected()
        data = message.get("bytes")
        if data is not None:
            if self.recorder is not None:
                self.recorder.add_caller(data)
            return AudioFrame(pcm=data)
        text = message.get("text")
        if text:
            try:
                return ControlMessage(data=json.loads(text))
            except json.JSONDecodeError:
                return ControlMessage(data={"type": "_invalid", "raw": text})
        return ControlMessage(data={"type": "_empty"})

    async def send_audio(self, turn_id: int, pcm: bytes) -> None:
        if self.recorder is not None:
            self.recorder.add_agent(pcm)
        await self._ws.send_bytes(struct.pack("<I", turn_id) + pcm)

    async def send_control(self, data: dict) -> None:
        await self._ws.send_text(json.dumps(data))

    async def close(self) -> None:
        try:
            await self._ws.close()
        except RuntimeError:
            pass  # already closed
