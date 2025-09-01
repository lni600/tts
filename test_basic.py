#!/usr/bin/env python3
"""
Basic test script to verify core functionality.
"""
import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.streaming_llm import DummyStreamingLLM
from realtime.tts_elevenlabs_ws import DummyTTSClient
from utils.audio import pcm16_to_wav_bytes, write_wav
from utils.zipper import build_conversation_zip


async def test_llm():
    """Test the dummy LLM."""
    print("🧠 Testing Dummy LLM...")
    llm = DummyStreamingLLM()
    
    tokens = []
    async for token in llm.stream_assistant_reply("Hello, how are you?"):
        tokens.append(token)
        print(f"  Token: {token}")
    
    print(f"✅ LLM generated {len(tokens)} tokens")
    return " ".join(tokens)


async def test_tts():
    """Test the dummy TTS."""
    print("🎤 Testing Dummy TTS...")
    tts = DummyTTSClient()
    
    if await tts.connect():
        print("  ✅ TTS connected")
        
        # Collect audio chunks
        chunks = []
        async for chunk in tts.audio_chunks():
            chunks.append(chunk)
            if len(chunks) >= 5:  # Limit for testing
                break
        
        print(f"  ✅ TTS generated {len(chunks)} audio chunks")
        return chunks
    else:
        print("  ❌ TTS connection failed")
        return []


def test_audio_utils():
    """Test audio utilities."""
    print("🔊 Testing Audio Utils...")
    
    # Test PCM16 to WAV conversion
    test_audio = b'\x00\x00\x7F\x7F' * 160  # 1 second of audio at 16kHz
    wav_bytes = pcm16_to_wav_bytes(test_audio, 16000, 1)
    
    if len(wav_bytes) > 0:
        print("  ✅ PCM16 to WAV conversion successful")
        
        # Test WAV writing
        try:
            write_wav([test_audio], "test_output.wav", 16000)
            print("  ✅ WAV file writing successful")
            os.remove("test_output.wav")  # Cleanup
        except Exception as e:
            print(f"  ❌ WAV file writing failed: {e}")
    else:
        print("  ❌ PCM16 to WAV conversion failed")


def test_zipper():
    """Test conversation zipper."""
    print("📦 Testing Conversation Zipper...")
    
    # Create test messages
    test_messages = [
        {
            "role": "user",
            "text": "Hello",
            "started_at": "2024-01-01T00:00:00",
            "ended_at": "2024-01-01T00:00:00"
        },
        {
            "role": "assistant",
            "text": "Hi there!",
            "audio_path": "test_audio.wav",
            "started_at": "2024-01-01T00:00:01",
            "ended_at": "2024-01-01T00:00:02"
        }
    ]
    
    try:
        zip_bytes, filename = build_conversation_zip(test_messages)
        print(f"  ✅ ZIP creation successful: {filename}")
        print(f"  📊 ZIP size: {len(zip_bytes)} bytes")
    except Exception as e:
        print(f"  ❌ ZIP creation failed: {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting Basic Functionality Tests\n")
    
    # Test LLM
    llm_text = await test_llm()
    print()
    
    # Test TTS
    audio_chunks = await test_tts()
    print()
    
    # Test audio utilities
    test_audio_utils()
    print()
    
    # Test zipper
    test_zipper()
    print()
    
    print("🎉 All tests completed!")
    print(f"📝 LLM generated text: {llm_text[:50]}...")
    print(f"🎵 TTS generated {len(audio_chunks)} audio chunks")


if __name__ == "__main__":
    asyncio.run(main())
