"""
Audio utilities for PCM16 processing and WAV file operations.
"""
import wave
import io
import numpy as np
from typing import List, Union
import base64


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """
    Convert raw PCM16 bytes to WAV format bytes.
    
    Args:
        pcm_bytes: Raw PCM16 audio data
        sample_rate: Audio sample rate (default: 16000)
        channels: Number of audio channels (default: 1 for mono)
    
    Returns:
        WAV format bytes
    """
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    
    return wav_buffer.getvalue()


def write_wav(chunks: List[bytes], path: str, sample_rate: int = 16000) -> None:
    """
    Write a list of PCM16 chunks to a WAV file.
    
    Args:
        chunks: List of PCM16 audio chunks
        path: Output file path
        sample_rate: Audio sample rate (default: 16000)
    """
    # Concatenate all chunks
    combined_audio = b''.join(chunks)
    
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(combined_audio)


def base64_to_pcm16(base64_string: str) -> bytes:
    """
    Convert base64 encoded audio to PCM16 bytes.
    
    Args:
        base64_string: Base64 encoded audio data
    
    Returns:
        PCM16 bytes
    """
    return base64.b64decode(base64_string)


def pcm16_to_numpy(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert PCM16 bytes to numpy array.
    
    Args:
        pcm_bytes: Raw PCM16 audio data
    
    Returns:
        Numpy array of audio samples
    """
    return np.frombuffer(pcm_bytes, dtype=np.int16)


def numpy_to_pcm16(audio_array: np.ndarray) -> bytes:
    """
    Convert numpy array to PCM16 bytes.
    
    Args:
        audio_array: Numpy array of audio samples
    
    Returns:
        PCM16 bytes
    """
    return audio_array.astype(np.int16).tobytes()


def calculate_frame_size(sample_rate: int, chunk_duration_ms: int) -> int:
    """
    Calculate the number of samples for a given duration.
    
    Args:
        sample_rate: Audio sample rate
        chunk_duration_ms: Duration in milliseconds
    
    Returns:
        Number of samples
    """
    return int(sample_rate * chunk_duration_ms / 1000)


def frame_size_to_bytes(frame_size: int, channels: int = 1) -> int:
    """
    Convert frame size in samples to bytes.
    
    Args:
        frame_size: Number of samples
        channels: Number of channels
    
    Returns:
        Size in bytes
    """
    return frame_size * channels * 2  # 2 bytes per sample (16-bit)
