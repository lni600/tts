"""
ElevenLabs WebSocket TTS client for realtime audio streaming.
"""
import asyncio
import json
import base64
import websockets
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
from datetime import datetime
import os


class ElevenLabsRealtimeClient:
    """ElevenLabs WebSocket TTS client for realtime streaming."""
    
    def __init__(self, voice_id: str, api_key: str, sample_rate: int = 16000):
        self.voice_id = voice_id
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.text_buffer = ""
        self.last_flush_time = 0
        self.flush_interval = 0.15  # 150ms flush interval
        
    async def connect(self) -> bool:
        """
        Connect to ElevenLabs WebSocket endpoint.
        
        Returns:
            True if connection successful
        """
        try:
            # ElevenLabs WebSocket URL
            ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input"
            
            # Connection headers
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(ws_url, extra_headers=headers)
            
            # Send initial configuration
            config_message = {
                "text": "",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                },
                "generation_config": {
                    "chunk_length_schedule": [50],
                    "temperature": 0.7
                },
                "xi_api_key": self.api_key
            }
            
            await self.websocket.send(json.dumps(config_message))
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("status") == "ok":
                self.connected = True
                return True
            else:
                print(f"Connection failed: {response_data}")
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    async def send_text_fragment(self, text: str) -> None:
        """
        Send text fragment to TTS service.
        
        Args:
            text: Text fragment to synthesize
        """
        if not self.connected or not self.websocket:
            return
            
        # Add to buffer
        self.text_buffer += text
        
        # Check if we should flush (time-based or punctuation-based)
        current_time = asyncio.get_event_loop().time()
        should_flush = (
            current_time - self.last_flush_time >= self.flush_interval or
            any(punct in text for punct in '.!?,:;')
        )
        
        if should_flush and self.text_buffer.strip():
            await self._flush_text_buffer()
    
    async def _flush_text_buffer(self) -> None:
        """Flush the text buffer to TTS service."""
        if not self.text_buffer.strip():
            return
            
        try:
            message = {
                "text": self.text_buffer,
                "try_trigger_generation": True
            }
            
            await self.websocket.send(json.dumps(message))
            self.text_buffer = ""
            self.last_flush_time = asyncio.get_event_loop().time()
            
        except Exception as e:
            print(f"Error flushing text buffer: {e}")
    
    async def audio_chunks(self) -> AsyncGenerator[bytes, None]:
        """
        Async generator yielding audio chunks as they arrive.
        
        Yields:
            PCM16 audio bytes
        """
        if not self.connected or not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if "audio" in data:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(data["audio"])
                        yield audio_bytes
                        
                    elif "isFinal" in data and data["isFinal"]:
                        # Final chunk, flush any remaining text
                        await self._flush_text_buffer()
                        
                except json.JSONDecodeError:
                    print(f"Invalid JSON message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            print(f"Error in audio stream: {e}")
            self.connected = False
    
    async def finalize(self) -> None:
        """Finalize the TTS generation."""
        if self.connected and self.websocket:
            # Flush any remaining text
            await self._flush_text_buffer()
            
            # Send end signal
            try:
                end_message = {
                    "text": "",
                    "try_trigger_generation": True
                }
                await self.websocket.send(json.dumps(end_message))
            except Exception as e:
                print(f"Error sending end signal: {e}")
    
    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
        self.connected = False


class DummyTTSClient:
    """Dummy TTS client for testing without ElevenLabs credentials."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.connected = True
        
    async def connect(self) -> bool:
        """Simulate connection."""
        return True
    
    async def send_text_fragment(self, text: str) -> None:
        """Simulate sending text (no-op)."""
        pass
    
    async def audio_chunks(self) -> AsyncGenerator[bytes, None]:
        """
        Generate dummy audio chunks.
        
        Yields:
            PCM16 audio bytes (sine wave)
        """
        # Generate a simple sine wave for testing
        frequency = 440  # A4 note
        duration_ms = 20  # 20ms chunks
        samples_per_chunk = int(self.sample_rate * duration_ms / 1000)
        
        chunk_count = 0
        while chunk_count < 50:  # Generate ~1 second of audio
            # Generate sine wave samples
            t = np.linspace(0, duration_ms/1000, samples_per_chunk, False)
            audio_samples = np.sin(2 * np.pi * frequency * t)
            
            # Convert to PCM16
            pcm16_samples = (audio_samples * 32767).astype(np.int16)
            audio_bytes = pcm16_samples.tobytes()
            
            yield audio_bytes
            
            chunk_count += 1
            await asyncio.sleep(duration_ms / 1000)  # Simulate real-time
    
    async def finalize(self) -> None:
        """Simulate finalization."""
        pass
    
    async def close(self) -> None:
        """Simulate closing."""
        self.connected = False


def create_tts_client(
    voice_id: str, 
    api_key: str, 
    use_dummy: bool = True,
    sample_rate: int = 16000
) -> ElevenLabsRealtimeClient | DummyTTSClient:
    """
    Factory function to create appropriate TTS client.
    
    Args:
        voice_id: ElevenLabs voice ID
        api_key: ElevenLabs API key
        use_dummy: Whether to use dummy TTS
        sample_rate: Audio sample rate
        
    Returns:
        TTS client instance
    """
    if use_dummy:
        return DummyTTSClient(sample_rate)
    else:
        return ElevenLabsRealtimeClient(voice_id, api_key, sample_rate)


async def test_tts_client():
    """Test function for TTS client."""
    client = DummyTTSClient()
    
    if await client.connect():
        print("Connected to TTS service")
        
        # Send some text
        await client.send_text_fragment("Hello, this is a test.")
        
        # Collect audio chunks
        chunk_count = 0
        async for chunk in client.audio_chunks():
            print(f"Received audio chunk {chunk_count}: {len(chunk)} bytes")
            chunk_count += 1
            if chunk_count >= 10:
                break
        
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_tts_client())
