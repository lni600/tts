"""
Speech-to-Text using OpenAI Realtime API.
"""
import asyncio
import json
import base64
import ssl
import websockets
from typing import Optional, Dict, Any, Callable
import logging
import time

logger = logging.getLogger(__name__)

class OpenAIRealtimeSTT:
    """OpenAI Realtime API Speech-to-Text client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview-2024-10-01"):
        """
        Initialize OpenAI Realtime STT client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for realtime API
        """
        self.api_key = api_key
        self.model = model
        self.websocket = None
        self.is_connected = False
        self.session_id = None
        self.on_transcription_callback: Optional[Callable[[str], None]] = None
        self.on_error_callback: Optional[Callable[[str], None]] = None
        self.audio_buffer = b""
        self.sample_rate = 24000  # OpenAI Realtime API uses 24kHz
        self.channels = 1
        self.bytes_per_sample = 2  # 16-bit
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 2  # seconds
        
        # Recording state
        self.is_recording = False
        self.recorded_audio = b""
        
    def set_transcription_callback(self, callback: Callable[[str], None]):
        """Set callback for transcription results."""
        self.on_transcription_callback = callback
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors."""
        self.on_error_callback = callback
    
    async def connect(self):
        """Connect to OpenAI Realtime API."""
        try:
            # Create WebSocket connection
            uri = "wss://api.openai.com/v1/realtime?model=" + self.model
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            logger.info("Connecting to OpenAI Realtime API...")
            self.websocket = await websockets.connect(
                uri, 
                additional_headers=headers,
                ssl=True,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,  # 1MB max message size
                compression=None
            )
            self.is_connected = True
            logger.info("Connected to OpenAI Realtime API")
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages())
            
            # Send session configuration
            await self._send_session_config()
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI Realtime API: {e}")
            self.is_connected = False
            if self.on_error_callback:
                self.on_error_callback(f"Connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from OpenAI Realtime API."""
        try:
            if self.websocket and self.is_connected:
                await self.websocket.close()
                self.is_connected = False
                logger.info("Disconnected from OpenAI Realtime API")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    async def reconnect(self):
        """Attempt to reconnect to the API."""
        for attempt in range(self.max_reconnect_attempts):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts}")
                await asyncio.sleep(self.reconnect_delay * (attempt + 1))  # Exponential backoff
                await self.connect()
                if self.is_connected:
                    logger.info("Successfully reconnected")
                    return True
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("All reconnection attempts failed")
        if self.on_error_callback:
            self.on_error_callback("Failed to reconnect after multiple attempts")
        return False
    
    async def _send_session_config(self):
        """Send session configuration."""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "You are a helpful assistant. Transcribe the user's speech accurately.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 4096
            }
        }
        
        await self.websocket.send(json.dumps(config))
        logger.info("Sent session configuration")
    
    async def _listen_for_messages(self):
        """Listen for messages from the API."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"WebSocket connection closed: {e}")
            self.is_connected = False
            # Attempt reconnection for unexpected closures
            if e.code != 1000:  # Not a normal closure
                logger.info("Attempting to reconnect...")
                await self.reconnect()
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            logger.info("Attempting to reconnect...")
            await self.reconnect()
        except ssl.SSLError as e:
            logger.error(f"SSL error: {e}")
            self.is_connected = False
            logger.info("Attempting to reconnect...")
            await self.reconnect()
        except ConnectionResetError as e:
            logger.error(f"Connection reset: {e}")
            self.is_connected = False
            logger.info("Attempting to reconnect...")
            await self.reconnect()
        except Exception as e:
            logger.error(f"Unexpected error in message listener: {e}")
            self.is_connected = False
            if self.on_error_callback:
                self.on_error_callback(f"Unexpected error: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages from the API."""
        message_type = data.get("type")
        
        if message_type == "session.created":
            self.session_id = data.get("session", {}).get("id")
            logger.info(f"Session created: {self.session_id}")
            
        elif message_type == "session.updated":
            logger.info("Session updated")
            
        elif message_type == "input_audio_buffer.speech_started":
            logger.info("Speech started")
            
        elif message_type == "input_audio_buffer.speech_stopped":
            logger.info("Speech stopped")
            
        elif message_type == "conversation.item.input_audio_transcription.completed":
            # Handle transcription result
            transcription = data.get("transcript", "")
            logger.info(f"Transcription received: '{transcription}'")
            if transcription and self.on_transcription_callback:
                logger.info("Calling transcription callback...")
                self.on_transcription_callback(transcription)
            else:
                logger.warning(f"No transcription callback or empty transcription: '{transcription}'")
            
        elif message_type == "error":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            logger.error(f"API Error: {error_msg}")
            if self.on_error_callback:
                self.on_error_callback(error_msg)
    
    async def send_audio_data(self, audio_bytes: bytes):
        """Send audio data to the API."""
        if not self.is_connected or not self.websocket:
            logger.warning("Not connected to API")
            return
        
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Failed to send audio data: {e}")
            if self.on_error_callback:
                self.on_error_callback(f"Failed to send audio: {e}")
    
    async def commit_audio_buffer(self):
        """Commit the current audio buffer for processing."""
        if not self.is_connected or not self.websocket:
            logger.warning("Not connected to API")
            return
        
        try:
            message = {
                "type": "input_audio_buffer.commit"
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info("Committed audio buffer")
            
        except Exception as e:
            logger.error(f"Failed to commit audio buffer: {e}")
            if self.on_error_callback:
                self.on_error_callback(f"Failed to commit audio: {e}")
    
    async def process_audio_data(self, audio_bytes: bytes):
        """Process audio data from WebRTC stream."""
        if not self.is_connected:
            logger.warning("Not connected to API")
            return
        
        try:
            # Add audio data to buffer
            self.audio_buffer += audio_bytes
            
            # Send audio data to API
            await self.send_audio_data(audio_bytes)
            
        except Exception as e:
            logger.error(f"Failed to process audio data: {e}")
            if self.on_error_callback:
                self.on_error_callback(f"Failed to process audio: {e}")
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, sample_rate: int = 24000) -> Dict[str, Any]:
        """
        Transcribe audio from bytes (synchronous wrapper).
        
        Args:
            audio_bytes: Audio data as bytes
            sample_rate: Sample rate of audio (should be 24000 for OpenAI Realtime)
            
        Returns:
            Dict with transcription results
        """
        # This is a synchronous wrapper - in practice, you'd want to use the async methods
        # for real-time streaming. This method is kept for compatibility.
        logger.warning("transcribe_audio_bytes is deprecated for realtime API. Use async methods instead.")
        
        return {
            "text": "",
            "language": "unknown",
            "segments": [],
            "success": False,
            "error": "Use async send_audio_data and commit_audio_buffer methods for realtime transcription"
        }
    
    def transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio from file (synchronous wrapper).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcription results
        """
        logger.warning("transcribe_audio_file is deprecated for realtime API. Use async methods instead.")
        
        return {
            "text": "",
            "language": "unknown",
            "segments": [],
            "success": False,
            "error": "Use async send_audio_data and commit_audio_buffer methods for realtime transcription"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "is_connected": self.is_connected,
            "session_id": self.session_id,
            "model": self.model,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bytes_per_sample": self.bytes_per_sample,
            "is_recording": self.is_recording,
            "recorded_audio_size": len(self.recorded_audio)
        }
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get recording statistics (alias for get_stats for compatibility)."""
        return self.get_stats()
    
    def start_recording(self):
        """Start recording audio (placeholder for browser-based recording)."""
        logger.info("Start recording requested - this should be handled by browser audio capture")
        self.is_recording = True
        self.recorded_audio = b""
        self.audio_buffer = b""  # Clear the audio buffer
        
    def stop_recording(self):
        """Stop recording audio."""
        logger.info("Stop recording requested")
        self.is_recording = False
        
    async def commit_audio_for_transcription(self) -> Dict[str, Any]:
        """
        Commit recorded audio for transcription.
        """
        logger.info("Commit audio for transcription requested")
        
        if not self.is_connected:
            return {
                "success": False,
                "error": "Not connected to OpenAI API"
            }
        
        if not self.audio_buffer:
            return {
                "success": False,
                "error": "No audio recorded"
            }
        
        try:
            logger.info(f"Committing {len(self.audio_buffer)} bytes of audio data")
            
            # Commit the buffer for transcription
            await self.commit_audio_buffer()
            
            return {
                "success": True,
                "message": "Audio committed for transcription"
            }
            
        except Exception as e:
            logger.error(f"Failed to commit audio: {e}")
            return {
                "success": False,
                "error": str(e)
            }

def create_openai_realtime_stt(api_key: str, model: str = "gpt-4o-realtime-preview-2024-10-01") -> OpenAIRealtimeSTT:
    """Create an OpenAI Realtime STT client."""
    return OpenAIRealtimeSTT(api_key=api_key, model=model)
