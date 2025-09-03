"""
Speech-to-Text service combining audio capture and OpenAI Realtime API transcription.
"""
import asyncio
import tempfile
import os
from typing import Optional, Dict, Any, Callable
import logging
from datetime import datetime

from .openai_realtime_stt import OpenAIRealtimeSTT
from .audio_receiver import AudioReceiverTrack

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service with audio capture and OpenAI Realtime API transcription."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview-2024-10-01"):
        """
        Initialize STT service.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI Realtime model to use
        """
        self.openai_stt = OpenAIRealtimeSTT(api_key=api_key, model=model)
        self.audio_receiver = None
        self.is_recording = False
        self.audio_buffer = b""
        self.on_transcription_callback: Optional[Callable[[str], None]] = None
        self.on_error_callback: Optional[Callable[[str], None]] = None
        self.is_connected = False
        
    def set_transcription_callback(self, callback: Callable[[str], None]):
        """Set callback for transcription results."""
        self.on_transcription_callback = callback
        self.openai_stt.set_transcription_callback(callback)
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors."""
        self.on_error_callback = callback
        self.openai_stt.set_error_callback(callback)
    
    async def connect(self):
        """Connect to OpenAI Realtime API."""
        try:
            await self.openai_stt.connect()
            self.is_connected = True
            logger.info("STT Service connected to OpenAI Realtime API")
        except Exception as e:
            logger.error(f"Failed to connect STT Service: {e}")
            self.is_connected = False
            if self.on_error_callback:
                self.on_error_callback(f"Connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from OpenAI Realtime API."""
        try:
            await self.openai_stt.disconnect()
            self.is_connected = False
            logger.info("STT Service disconnected from OpenAI Realtime API")
        except Exception as e:
            logger.error(f"Error disconnecting STT Service: {e}")
    
    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        if not self.is_connected:
            logger.warning("Not connected to OpenAI Realtime API")
            return
        
        self.is_recording = True
        self.audio_buffer = b""
        logger.info("Started recording")
    
    def stop_recording(self):
        """Stop audio recording."""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return
        
        self.is_recording = False
        logger.info("Stopped recording")
    
    async def process_audio_data(self, audio_data: bytes):
        """Process incoming audio data and send to OpenAI Realtime API."""
        if self.is_recording and self.is_connected:
            self.audio_buffer += audio_data
            # Send audio data to OpenAI Realtime API
            await self.openai_stt.send_audio_data(audio_data)
    
    async def commit_audio_for_transcription(self):
        """Commit current audio buffer for transcription."""
        if not self.is_connected:
            logger.warning("Not connected to OpenAI Realtime API")
            return {
                "text": "",
                "success": False,
                "error": "Not connected to API"
            }
        
        try:
            await self.openai_stt.commit_audio_buffer()
            return {
                "text": "",
                "success": True,
                "message": "Audio committed for transcription"
            }
        except Exception as e:
            error_msg = f"Failed to commit audio: {e}"
            logger.error(error_msg)
            return {
                "text": "",
                "success": False,
                "error": error_msg
            }
    
    def transcribe_current_audio(self) -> Dict[str, Any]:
        """Transcribe current audio buffer (deprecated for realtime API)."""
        logger.warning("transcribe_current_audio is deprecated for realtime API. Use async methods instead.")
        return {
            "text": "",
            "success": False,
            "error": "Use async commit_audio_for_transcription method for realtime transcription"
        }
    
    def transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio from file (deprecated for realtime API)."""
        logger.warning("transcribe_audio_file is deprecated for realtime API. Use async methods instead.")
        return {
            "text": "",
            "success": False,
            "error": "Use async methods for realtime transcription"
        }
    
    def save_audio_buffer(self, file_path: str) -> bool:
        """Save current audio buffer to file."""
        try:
            with open(file_path, 'wb') as f:
                f.write(self.audio_buffer)
            logger.info(f"Saved audio buffer to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save audio buffer: {e}")
            return False
    
    def clear_audio_buffer(self):
        """Clear audio buffer."""
        self.audio_buffer = b""
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        return {
            "is_recording": self.is_recording,
            "is_connected": self.is_connected,
            "buffer_size_bytes": len(self.audio_buffer),
            "buffer_duration_estimate": len(self.audio_buffer) / (24000 * 2),  # 24kHz estimate
            "model": self.openai_stt.model,
            "session_id": self.openai_stt.session_id
        }

def create_stt_service(api_key: str, model: str = "gpt-4o-realtime-preview-2024-10-01") -> STTService:
    """Create STT service with OpenAI Realtime API."""
    return STTService(api_key=api_key, model=model)
