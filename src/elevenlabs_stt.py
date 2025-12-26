"""
ElevenLabs Real-Time Streaming STT

Uses ElevenLabs Python SDK's Realtime Speech-to-Text API.
Ultra-low latency (~150ms) transcription with automatic language detection.

API Reference: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime

Input: PCM 16-bit audio buffer (bytes)
Output: STT events (stt_chunk for partials, stt_output for final transcripts)
"""

import asyncio
import base64
import os
from typing import AsyncIterator, Optional

from elevenlabs.client import ElevenLabs
from elevenlabs import RealtimeEvents, AudioFormat
from elevenlabs.realtime import RealtimeAudioOptions, CommitStrategy

from events import STTChunkEvent, STTEvent, STTOutputEvent
from logger import logger


class ElevenLabsSTT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        sample_rate: int = 16000,
        language_code: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.client = ElevenLabs(api_key=self.api_key)
        self.sample_rate = sample_rate
        self.language_code = language_code
        self._session = None
        self._event_queue = asyncio.Queue()
        self._close_event = asyncio.Event()

    async def receive_events(self) -> AsyncIterator[STTEvent]:
        """Receive transcription events from ElevenLabs."""
        # Ensure connection is established (lazy connection if not already)
        await self._ensure_connection()

        while not self._close_event.is_set():
            try:
                # Wait for next event with a timeout to allow checking close_event
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio chunk to ElevenLabs for transcription."""
        retries = 1
        for attempt in range(retries + 1):
            try:
                await self._ensure_connection()
                if self._session:
                    audio_base64 = base64.b64encode(audio_chunk).decode("utf-8")
                    await self._session.send({
                        "audio_base_64": audio_base64,
                        "sample_rate": self.sample_rate
                    })
                break
            except Exception as e:
                logger.warning(f"[Warning] Failed to send audio chunk (attempt {attempt+1}/{retries+1}): {e}")
                # Force reset session on error to trigger reconnect next time
                self._session = None
                if attempt == retries:
                    # Don't raise, just log to prevent stream crash on single chunk failure
                    logger.error("[Error] Could not send audio chunk after retries.")

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._is_closing = True
        self._close_event.set()
        if self._session:
            try:
                # Close the session if possible
                if hasattr(self._session, "close"):
                    await self._session.close()
            except Exception:
                pass
        self._session = None

    async def _ensure_connection(self):
        """Ensure WebSocket connection is established."""
        if self._session:
            return

        if self._close_event.is_set() and not hasattr(self, "_is_closing"):
             # Reset close event if we are reconnecting after a drop
             self._close_event.clear()
        
        # Guard against reusing a closed event loop if strictly checking logic, 
        # but here we just ensure we can create a new session.

        # Build options using RealtimeAudioOptions
        options_kwargs = {
            "model_id": "scribe_v2_realtime",
            "audio_format": AudioFormat.PCM_16000,
            "sample_rate": self.sample_rate,
            "include_timestamps": False,
            "commit_strategy": CommitStrategy.VAD,
            "vad_silence_threshold_secs": 1.0,
            "vad_threshold": 0.3,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 500,
        }
        if self.language_code:
            options_kwargs["language_code"] = self.language_code

        options = RealtimeAudioOptions(**options_kwargs)

        try:
            # Connect using the SDK
            self._session = await self.client.speech_to_text.realtime.connect(options=options)
            logger.info("[STT] Connected to ElevenLabs")
        except Exception as e:
            logger.error(f"[STT] Connection failed: {e}")
            self._session = None
            return

        # Register event handlers
        def on_partial(data):
            text = data.get("text", "")
            if text:
                logger.debug(f"[DEBUG] STT Partial: {text}")
                self._event_queue.put_nowait(STTChunkEvent.create(text))

        def on_committed(data):
            text = data.get("text", "")
            if text:
                logger.debug(f"[DEBUG] STT Committed: {text}")
                self._event_queue.put_nowait(STTOutputEvent.create(text))
        
        def on_error(error):
            logger.error(f"[STT] Error: {error}")
            self._session = None
            # Do NOT set close_event, allowing reconnect

        def on_close():
            logger.info("[STT] Session closed")
            self._session = None
            if getattr(self, "_is_closing", False):
                self._close_event.set()

        self._session.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
        self._session.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
        self._session.on(RealtimeEvents.ERROR, on_error)
        self._session.on(RealtimeEvents.CLOSE, on_close)

