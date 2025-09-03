"""
WebRTC audio receiver for capturing microphone input for OpenAI Realtime API.
"""
import asyncio
import av
import numpy as np
from aiortc import MediaStreamTrack
from typing import Optional, Callable, List
import time
import logging

logger = logging.getLogger(__name__)

class AudioReceiverTrack(MediaStreamTrack):
    """MediaStreamTrack for receiving audio from microphone for OpenAI Realtime API."""
    
    kind = "audio"
    
    def __init__(self, on_audio_data: Optional[Callable[[bytes], None]] = None):
        super().__init__()
        self.on_audio_data = on_audio_data
        self.audio_buffer = b""
        self.sample_rate = 24000  # OpenAI Realtime API uses 24kHz
        self.channels = 1
        self.bytes_per_sample = 2  # 16-bit PCM
        self.frame_count = 0
        self.start_time = time.time()
        
    async def recv(self) -> av.AudioFrame:
        """Receive audio frame from WebRTC."""
        try:
            # This is a receiver track, so we don't actually receive frames here
            # The actual audio data comes through the WebRTC callback
            await asyncio.sleep(0.1)
            return None
        except Exception as e:
            logger.error(f"Error in AudioReceiverTrack.recv: {e}")
            return None
    
    def process_audio_frame(self, frame: av.AudioFrame):
        """Process incoming audio frame for OpenAI Realtime API."""
        try:
            # Convert frame to bytes
            audio_data = frame.to_ndarray()
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Resample to 24kHz if needed (OpenAI Realtime API requirement)
            if frame.sample_rate != self.sample_rate:
                # Simple resampling - in production, you might want to use a proper resampling library
                ratio = self.sample_rate / frame.sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_data.tobytes()
            
            # Add to buffer
            self.audio_buffer += audio_bytes
            
            # Call callback if provided
            if self.on_audio_data:
                self.on_audio_data(audio_bytes)
                
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
    
    def get_audio_buffer(self) -> bytes:
        """Get current audio buffer."""
        return self.audio_buffer
    
    def clear_audio_buffer(self):
        """Clear audio buffer."""
        self.audio_buffer = b""
    
    def get_stats(self) -> dict:
        """Get receiver statistics."""
        elapsed = time.time() - self.start_time
        return {
            "frames_received": self.frame_count,
            "buffer_size_bytes": len(self.audio_buffer),
            "elapsed_time": elapsed,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }
