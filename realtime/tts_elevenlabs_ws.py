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
import io
import subprocess
import tempfile


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
        
        # Audio processing - no buffering needed
        self.audio_buffer = b""  # Initialize audio buffer
        
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
            self.websocket = await websockets.connect(ws_url, additional_headers=headers)
            
            # For the new ElevenLabs WebSocket API, we don't need to send an initial config
            # The connection is established and ready to receive text immediately
            self.connected = True
            return True
                
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
    
    def _convert_audio_to_pcm16(self, audio_bytes: bytes) -> bytes:
        """
        Convert audio bytes to PCM16 format with intelligent noise reduction.
        
        Args:
            audio_bytes: Raw audio bytes from ElevenLabs
            
        Returns:
            PCM16 audio bytes
        """
        try:
            # First, try to interpret as PCM16 directly (most likely format for ElevenLabs)
            if len(audio_bytes) % 2 == 0:
                try:
                    pcm16_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    if len(pcm16_array) > 0:
                        # Check if this looks like valid PCM16 audio (not all zeros or extreme values)
                        if np.any(pcm16_array != 0) and np.all(np.abs(pcm16_array) <= 32767):
                            # Apply gentle noise reduction to preserve speech
                            cleaned_audio = self._reduce_noise_pcm16(pcm16_array)
                            # Direct PCM16 processing successful
                            return cleaned_audio.tobytes()
                except Exception as e:
                    # Direct PCM16 processing failed
                    pass
            
            # If not PCM16, try ffmpeg conversion with compatible filters
            
            # Try different output formats with compatible filters
            output_formats = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
            
            for ext in output_formats:
                try:
                    # Create temporary files
                    input_path = f"temp_input_{hash(audio_bytes) % 10000}{ext}"
                    output_path = f"temp_output_{hash(audio_bytes) % 10000}.wav"
                    
                    # Write input audio
                    with open(input_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Use compatible ffmpeg filters for your version
                    cmd = [
                        'ffmpeg', '-y',  # Overwrite output
                        '-i', input_path,
                        '-f', 'wav',
                        '-acodec', 'pcm_s16le',  # 16-bit PCM
                        '-ar', '16000',  # 16kHz sample rate
                        '-ac', '1',      # Mono
                        '-af', 'highpass=f=80,lowpass=f=8000,volume=1.2',  # Simple, compatible filters
                        output_path
                    ]
                    
                    # Run ffmpeg
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0 and os.path.exists(output_path):
                        # Read the converted audio
                        with open(output_path, 'rb') as f:
                            converted_audio = f.read()
                        
                        # Verify it's valid PCM16
                        if len(converted_audio) > 44 and len(converted_audio) % 2 == 0:  # Skip WAV header
                            pcm16_data = converted_audio[44:]  # Remove WAV header
                            if len(pcm16_data) > 0:
                                print(f"ffmpeg conversion successful with {ext}: {len(pcm16_data)} bytes")
                                return pcm16_data
                        else:
                            print(f"ffmpeg output invalid: {len(converted_audio)} bytes")
                    else:
                        print(f"ffmpeg conversion failed with {ext}: {result.stderr[:100]}...")
                        
                except subprocess.TimeoutExpired:
                    print(f"ffmpeg conversion timed out with {ext}")
                except Exception as e:
                    print(f"ffmpeg conversion error with {ext}: {e}")
                finally:
                    # Clean up temporary files
                    try:
                        os.unlink(input_path)
                        os.unlink(output_path)
                    except:
                        pass
                        
            # If ffmpeg completely failed, try intelligent format detection
            print("ffmpeg conversion failed, trying intelligent format detection...")
            
            # Try to detect the actual audio format based on content analysis
            audio_formats = [
                ('float32', np.float32, 4),
                ('int16', np.int16, 2),
                ('int32', np.int32, 4)
            ]
            
            for format_name, dtype, bytes_per_sample in audio_formats:
                try:
                    # Check if buffer size is compatible
                    if len(audio_bytes) % bytes_per_sample == 0:
                        audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                        if len(audio_array) > 0:
                            # Analyze the audio content to see if it looks like speech
                            if dtype == np.float32:
                                # Check if values are in reasonable range for speech
                                if np.all(np.abs(audio_array) <= 1.0) and np.std(audio_array) > 0.01:
                                    print(f"Detected valid {format_name} speech audio: {len(audio_array)} samples")
                                    # Convert to PCM16 with gentle processing
                                    pcm16_array = (audio_array * 32767).astype(np.int16)
                                    return pcm16_array.tobytes()
                            elif dtype == np.int16:
                                # Check if values are in reasonable range for speech
                                if np.all(np.abs(audio_array) <= 32767) and np.std(audio_array) > 100:
                                    print(f"Detected valid {format_name} speech audio: {len(audio_array)} samples")
                                    # Apply gentle noise reduction
                                    cleaned_audio = self._reduce_noise_pcm16(audio_array)
                                    return cleaned_audio.tobytes()
                            elif dtype == np.int32:
                                # Check if values are in reasonable range for speech
                                if np.all(np.abs(audio_array) <= 2147483647) and np.std(audio_array) > 1000:
                                    print(f"Detected valid {format_name} speech audio: {len(audio_array)} samples")
                                    # Convert to PCM16 with gentle processing
                                    pcm16_array = (audio_array // 65536).astype(np.int16)  # Scale down
                                    return pcm16_array.tobytes()
                    else:
                        print(f"{format_name} buffer size {len(audio_bytes)} is not a multiple of {bytes_per_sample}")
                except Exception as e:
                    print(f"{format_name} processing failed: {e}")
                    continue
            
            # If all else fails, try to interpret as raw PCM with minimal processing
            print("Trying minimal raw audio processing...")
            try:
                # Assume it might be raw PCM and try to make it work
                if len(audio_bytes) % 2 == 0:
                    # Force as int16 and clip to valid range
                    raw_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    # Clip to valid range
                    clipped_array = np.clip(raw_array, -32767, 32767)
                    print(f"Raw PCM processing: {len(clipped_array)} samples")
                    return clipped_array.tobytes()
            except Exception as e:
                print(f"Raw PCM processing failed: {e}")
            
            # Final fallback: generate clean silence
            print("Could not convert audio, generating clean silence")
            return self._generate_clean_silence()
            
        except Exception as e:
            print(f"Error converting audio to PCM16: {e}")
            # Generate clean silence as fallback
            return self._generate_clean_silence()
    
    def _reduce_noise_pcm16(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply gentle noise reduction to PCM16 audio while preserving speech.
        
        Args:
            audio_array: PCM16 audio array
            
        Returns:
            Cleaned PCM16 audio array
        """
        try:
            # Convert to float for processing
            audio_float = audio_array.astype(np.float64) / 32767.0
            
            # Apply gentle noise reduction
            cleaned_float = self._reduce_noise_float(audio_float)
            
            # Convert back to PCM16
            cleaned_pcm16 = (cleaned_float * 32767).astype(np.int16)
            
            return cleaned_pcm16
            
        except Exception as e:
            print(f"Error in PCM16 noise reduction: {e}")
            return audio_array
    
    def _reduce_noise_float(self, audio_float: np.ndarray) -> np.ndarray:
        """
        Apply gentle noise reduction to float audio while preserving speech.
        
        Args:
            audio_float: Float audio array in [-1, 1] range
            
        Returns:
            Cleaned float audio array
        """
        try:
            # Simple, gentle noise reduction that preserves speech
            # 1. High-pass filter to remove very low frequency noise
            cleaned = self._simple_highpass_filter(audio_float)
            
            # 2. Gentle noise gate (only remove very quiet parts)
            cleaned = self._gentle_noise_gate(cleaned)
            
            # 3. Light normalization
            cleaned = np.clip(cleaned, -1.0, 1.0)
            
            return cleaned
            
        except Exception as e:
            print(f"Error in float noise reduction: {e}")
            return audio_float
    
    def _simple_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple high-pass filter to remove very low frequency noise.
        
        Args:
            audio: Float audio array
            
        Returns:
            Filtered audio array
        """
        try:
            # Simple 1st order high-pass filter
            # This preserves speech while removing very low frequency noise
            alpha = 0.95  # Gentle filter
            filtered = np.zeros_like(audio)
            
            for i in range(1, len(audio)):
                filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
            
            return filtered
            
        except Exception as e:
            print(f"Error in high-pass filter: {e}")
            return audio
    
    def _gentle_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Gentle noise gate that only removes very quiet parts.
        
        Args:
            audio: Float audio array
            
        Returns:
            Gated audio array
        """
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            # Only gate if audio is very quiet (preserve speech)
            threshold = max(0.01, rms * 0.1)  # Very gentle threshold
            
            # Apply gate
            gated = np.where(np.abs(audio) < threshold, 0, audio)
            
            return gated
            
        except Exception as e:
            print(f"Error in noise gate: {e}")
            return audio
    
    # Crossfade method removed - not needed for immediate audio processing
    
    # Buffer processing removed - audio is now processed and yielded immediately
    
    def _is_essentially_silence(self, audio_bytes: bytes) -> bool:
        """
        Check if audio bytes represent essentially silence.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            True if audio is essentially silence
        """
        try:
            # Try different data types
            silence_detected = []
            
            for dtype in [np.int16, np.float32, np.float64]:
                try:
                    audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                    if len(audio_array) > 0:
                        # Calculate RMS (Root Mean Square) for amplitude
                        # Use a safer approach to avoid overflow
                        audio_float = audio_array.astype(np.float64)
                        
                        # Handle potential overflow by clipping
                        if dtype == np.int16:
                            # For int16, normalize to [-1, 1] range first
                            audio_normalized = np.clip(audio_float / 32767.0, -1.0, 1.0)
                            threshold = 0.005  # 0.5% of full scale for int16
                        else:
                            # For float types, use directly
                            audio_normalized = np.clip(audio_float, -1.0, 1.0)
                            threshold = 0.005  # 0.5% of full scale for float
                        
                        # Calculate RMS safely
                        rms = np.sqrt(np.mean(audio_normalized ** 2))
                        
                        # Check if this data type shows silence
                        is_silence = rms < threshold
                        silence_detected.append(is_silence)
                        
                except Exception as e:
                    print(f"Error processing {dtype}: {e}")
                    continue
            
            # If we have any valid results, check if ALL show silence
            if silence_detected:
                return all(silence_detected)
            
            return False
        except Exception as e:
            print(f"Error in silence detection: {e}")
            return False
    
    def _generate_clean_silence(self) -> bytes:
        """
        Generate clean, properly formatted silence.
        
        Returns:
            Clean PCM16 silence bytes
        """
        # Generate 20ms of clean silence at 16kHz
        silence_samples = np.zeros(320, dtype=np.int16)  # 16000 * 0.02 = 320 samples
        return silence_samples.tobytes()
    
    def _process_audio_buffer(self) -> Optional[bytes]:
        """Process audio buffer and return processed audio data."""
        if not self.audio_buffer:
            return None
        
        # For now, just return the buffer and clear it
        # This is a simplified implementation
        processed_audio = self.audio_buffer
        self.audio_buffer = b""
        return processed_audio
    
    async def audio_chunks(self) -> AsyncGenerator[bytes, None]:
        """
        Async generator yielding audio chunks as they arrive.
        Applies intelligent noise reduction and buffering for smooth speech.
        
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
                    if "audio" in data and data["audio"] is not None:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(data["audio"])
                        
                        # Convert to PCM16 format with intelligent noise reduction
                        pcm16_bytes = self._convert_audio_to_pcm16(audio_bytes)
                        
                        # Yield the processed audio immediately (no buffering loop)
                        if pcm16_bytes and len(pcm16_bytes) > 0:
                            yield pcm16_bytes
                        
                    elif "isFinal" in data and data["isFinal"]:
                        # Final chunk - we're done
                        break
                        
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
            
            # Process any remaining audio in buffer
            while self.audio_buffer:
                try:
                    processed_audio = self._process_audio_buffer()
                    if processed_audio:
                        # Yield the final processed audio
                        # Note: This is a bit of a hack since we're not in an async generator context
                        # The actual yielding happens in audio_chunks when isFinal is received
                        pass
                except Exception as e:
                    print(f"Error processing final audio buffer: {e}")
                    break
            
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
    
    def __init__(self, voice_id: str = "dummy", api_key: str = "dummy", sample_rate: int = 16000):
        self.voice_id = voice_id
        self.api_key = api_key
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
        pass


def create_tts_client(
    voice_id: str, 
    api_key: str, 
    use_dummy: bool = True,
    sample_rate: int = 16000
) -> ElevenLabsRealtimeClient | DummyTTSClient:
    """
    Create a TTS client.
    
    Args:
        voice_id: ElevenLabs voice ID
        api_key: ElevenLabs API key
        use_dummy: Whether to use dummy TTS for testing
        sample_rate: Target audio sample rate
        
    Returns:
        TTS client instance
    """
    if use_dummy:
        return DummyTTSClient(voice_id=voice_id, api_key=api_key, sample_rate=sample_rate)
    else:
        return ElevenLabsRealtimeClient(voice_id=voice_id, api_key=api_key, sample_rate=sample_rate)


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
