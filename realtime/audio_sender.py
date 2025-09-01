"""
WebRTC audio sender for realtime PCM16 audio streaming.
"""
import asyncio
import numpy as np
import av
from aiortc import MediaStreamTrack
from typing import Optional, List
import time


class PCMTrack(MediaStreamTrack):
    """MediaStreamTrack for PCM16 audio streaming."""
    
    kind = "audio"
    
    def __init__(self, queue: asyncio.Queue[bytes], sample_rate: int = 16000, chunk_duration_ms: int = 20):
        super().__init__()
        self.queue = queue
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        
        # Calculate frame size
        self.samples_per_frame = int(sample_rate * chunk_duration_ms / 1000)
        self.bytes_per_frame = self.samples_per_frame * 2  # 16-bit = 2 bytes
        
        # Audio buffer for partial frames
        self.audio_buffer = b""
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Jitter buffer settings
        self.max_buffer_size = 3  # Maximum frames to buffer
        self.silence_threshold = 0.1  # Seconds of silence before generating silence
        
    async def recv(self) -> av.AudioFrame:
        """
        Generate audio frame from PCM16 data.
        
        Returns:
            AudioFrame for WebRTC streaming
        """
        try:
            # Try to get audio data from queue
            try:
                # Non-blocking get with timeout
                audio_chunk = await asyncio.wait_for(self.queue.get(), timeout=0.05)
                self.audio_buffer += audio_chunk
            except asyncio.TimeoutError:
                # No audio data available, check if we should generate silence
                current_time = time.time()
                if current_time - self.last_frame_time > self.silence_threshold:
                    # Generate silence frame
                    return self._create_silence_frame()
                else:
                    # Reuse last frame or generate minimal silence
                    return self._create_silence_frame()
            
            # Process audio buffer
            frame = await self._process_audio_buffer()
            if frame:
                self.frame_count += 1
                self.last_frame_time = time.time()
                return frame
            else:
                # Buffer not full enough, generate silence
                return self._create_silence_frame()
                
        except Exception as e:
            print(f"Error in PCMTrack.recv: {e}")
            return self._create_silence_frame()
    
    async def _process_audio_buffer(self) -> Optional[av.AudioFrame]:
        """
        Process audio buffer and create frame if enough data.
        
        Returns:
            AudioFrame if buffer has enough data, None otherwise
        """
        # Check if we have enough data for a complete frame
        if len(self.audio_buffer) >= self.bytes_per_frame:
            # Extract frame data
            frame_data = self.audio_buffer[:self.bytes_per_frame]
            self.audio_buffer = self.audio_buffer[self.bytes_per_frame:]
            
            # Convert to numpy array
            audio_array = np.frombuffer(frame_data, dtype=np.int16)
            
            # Ensure correct shape
            if len(audio_array) != self.samples_per_frame:
                # Pad or truncate if necessary
                if len(audio_array) < self.samples_per_frame:
                    # Pad with zeros
                    padding = np.zeros(self.samples_per_frame - len(audio_array), dtype=np.int16)
                    audio_array = np.concatenate([audio_array, padding])
                else:
                    # Truncate
                    audio_array = audio_array[:self.samples_per_frame]
            
            # Create AudioFrame
            frame = av.AudioFrame.from_ndarray(
                audio_array.reshape(-1, 1),  # Reshape to (samples, channels)
                format="s16",  # 16-bit signed
                layout="mono"
            )
            
            # Set frame properties
            frame.sample_rate = self.sample_rate
            frame.pts = self.frame_count * self.samples_per_frame
            frame.time_base = av.time_base(1, self.sample_rate)
            
            return frame
        
        return None
    
    def _create_silence_frame(self) -> av.AudioFrame:
        """
        Create a silence frame for continuity.
        
        Returns:
            Silence AudioFrame
        """
        # Generate silence samples
        silence_samples = np.zeros(self.samples_per_frame, dtype=np.int16)
        
        # Create AudioFrame
        frame = av.AudioFrame.from_ndarray(
            silence_samples.reshape(-1, 1),
            format="s16",
            layout="mono"
        )
        
        # Set frame properties
        frame.sample_rate = self.sample_rate
        frame.pts = self.frame_count * self.samples_per_frame
        frame.time_base = av.time_base(1, self.sample_rate)
        
        return frame
    
    def stop(self):
        """Stop the track."""
        super().stop()


def get_track(queue: asyncio.Queue[bytes], sample_rate: int = 16000, chunk_duration_ms: int = 20) -> PCMTrack:
    """
    Create a PCMTrack from an audio queue.
    
    Args:
        queue: Queue containing PCM16 audio chunks
        sample_rate: Audio sample rate (default: 16000)
        chunk_duration_ms: Duration of each chunk in milliseconds (default: 20)
    
    Returns:
        PCMTrack instance
    """
    return PCMTrack(queue, sample_rate, chunk_duration_ms)


class AudioQueueManager:
    """Manages audio queues for multiple tracks."""
    
    def __init__(self):
        self.queues: List[asyncio.Queue[bytes]] = []
        self.active = True
    
    def create_queue(self) -> asyncio.Queue[bytes]:
        """
        Create a new audio queue.
        
        Returns:
            New audio queue
        """
        queue = asyncio.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.queues.append(queue)
        return queue
    
    def put_audio(self, audio_chunk: bytes) -> None:
        """
        Put audio chunk in all active queues.
        
        Args:
            audio_chunk: PCM16 audio data
        """
        if not self.active:
            return
            
        # Put in all queues
        for queue in self.queues:
            try:
                # Non-blocking put
                queue.put_nowait(audio_chunk)
            except asyncio.QueueFull:
                # Remove full queue
                self.queues.remove(queue)
    
    def clear_queues(self) -> None:
        """Clear all audio queues."""
        for queue in self.queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
    
    def stop(self) -> None:
        """Stop the audio manager."""
        self.active = False
        self.clear_queues()


async def test_audio_track():
    """Test function for audio track."""
    # Create a test queue
    queue = asyncio.Queue()
    
    # Create track
    track = get_track(queue)
    
    # Put some test audio data
    test_audio = np.random.randint(-32768, 32767, 320, dtype=np.int16).tobytes()
    await queue.put(test_audio)
    
    # Get frame
    frame = await track.recv()
    print(f"Generated frame: {frame}")
    
    track.stop()


if __name__ == "__main__":
    asyncio.run(test_audio_track())
