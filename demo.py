#!/usr/bin/env python3
"""
Demo script showing core functionality without external dependencies.
"""
import asyncio
import json
import zipfile
import io
from datetime import datetime


class SimpleDummyLLM:
    """Simple dummy LLM for demo."""
    
    async def stream_response(self, prompt: str):
        """Stream a simple response."""
        responses = [
            "Hello! I'm a demo chatbot. ",
            "You asked: ",
            f"'{prompt}'. ",
            "This is a demonstration of streaming text. ",
            "Each piece appears as it's generated. ",
            "In the real app, this would be connected to an LLM API. ",
            "Thanks for testing!"
        ]
        
        for response in responses:
            yield response
            await asyncio.sleep(0.1)  # Simulate processing time


class SimpleDummyTTS:
    """Simple dummy TTS for demo."""
    
    async def generate_audio(self, text: str):
        """Generate dummy audio data."""
        # Simulate audio generation
        audio_chunks = []
        for i in range(5):
            # Simulate PCM16 audio chunk (320 samples at 16kHz = 640 bytes)
            chunk = b'\x00\x00' * 320  # Silent audio
            audio_chunks.append(chunk)
            await asyncio.sleep(0.05)
        
        return audio_chunks


class SimpleAudioProcessor:
    """Simple audio processor for demo."""
    
    @staticmethod
    def create_wav_header(sample_rate=16000, channels=1, bits_per_sample=16):
        """Create a simple WAV header."""
        # WAV file header (44 bytes)
        header = bytearray(44)
        
        # RIFF header
        header[0:4] = b'RIFF'
        header[4:8] = (36).to_bytes(4, 'little')  # File size - 8
        header[8:12] = b'WAVE'
        
        # fmt chunk
        header[12:16] = b'fmt '
        header[16:20] = (16).to_bytes(4, 'little')  # fmt chunk size
        header[20:22] = (1).to_bytes(2, 'little')   # Audio format (PCM)
        header[22:24] = channels.to_bytes(2, 'little')
        header[24:28] = sample_rate.to_bytes(4, 'little')
        header[28:32] = (sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little')  # Byte rate
        header[32:34] = (channels * bits_per_sample // 8).to_bytes(2, 'little')  # Block align
        header[34:36] = bits_per_sample.to_bytes(2, 'little')
        
        # data chunk
        header[36:40] = b'data'
        header[40:44] = (0).to_bytes(4, 'little')  # Data size (will be updated)
        
        return header
    
    @staticmethod
    def create_wav_file(audio_chunks, sample_rate=16000, channels=1):
        """Create a WAV file from audio chunks."""
        # Calculate total audio size
        total_audio_size = sum(len(chunk) for chunk in audio_chunks)
        
        # Create header
        header = SimpleAudioProcessor.create_wav_header(sample_rate, channels)
        
        # Update data size in header
        header[40:44] = total_audio_size.to_bytes(4, 'little')
        
        # Combine header and audio
        wav_data = header + b''.join(audio_chunks)
        
        return wav_data


class SimpleZipper:
    """Simple conversation zipper for demo."""
    
    @staticmethod
    def create_conversation_zip(messages, audio_files):
        """Create a ZIP file with conversation data."""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add transcript
            transcript = {
                "conversation_id": f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.now().isoformat(),
                "messages": messages
            }
            
            zip_file.writestr("transcript.json", json.dumps(transcript, indent=2))
            
            # Add audio files
            for i, (filename, audio_data) in enumerate(audio_files):
                zip_file.writestr(f"audio/{filename}", audio_data)
            
            # Add README
            readme = f"""Demo Conversation Export
Generated: {datetime.now().isoformat()}
Total Messages: {len(messages)}

This is a demonstration export.
Audio files are dummy data for testing.
"""
            zip_file.writestr("README.txt", readme)
        
        return zip_buffer.getvalue()


async def demo_conversation():
    """Demonstrate a full conversation flow."""
    print("🎤 Streamlit TTS Chatbot Demo\n")
    
    # Initialize components
    llm = SimpleDummyLLM()
    tts = SimpleDummyTTS()
    audio_processor = SimpleAudioProcessor()
    zipper = SimpleZipper()
    
    # Simulate conversation
    conversation = []
    audio_files = []
    
    # User message
    user_prompt = "Hello, how are you today?"
    print(f"👤 User: {user_prompt}")
    
    user_message = {
        "role": "user",
        "text": user_prompt,
        "timestamp": datetime.now().isoformat()
    }
    conversation.append(user_message)
    
    # Assistant response
    print("\n🤖 Assistant (streaming):")
    assistant_text = ""
    
    async for token in llm.stream_response(user_prompt):
        assistant_text += token
        print(f"  {token}", end="", flush=True)
    
    print("\n")
    
    # Generate audio for assistant response
    print("🎵 Generating audio...")
    audio_chunks = await tts.generate_audio(assistant_text)
    
    # Create WAV file
    wav_data = audio_processor.create_wav_file(audio_chunks)
    audio_filename = f"assistant_{datetime.now().strftime('%H%M%S')}.wav"
    audio_files.append((audio_filename, wav_data))
    
    print(f"✅ Audio generated: {audio_filename} ({len(wav_data)} bytes)")
    
    # Add assistant message to conversation
    assistant_message = {
        "role": "assistant",
        "text": assistant_text,
        "audio_file": audio_filename,
        "timestamp": datetime.now().isoformat()
    }
    conversation.append(assistant_message)
    
    # Create ZIP export
    print("\n📦 Creating conversation export...")
    zip_data = zipper.create_conversation_zip(conversation, audio_files)
    
    print(f"✅ ZIP created: {len(zip_data)} bytes")
    
    # Show conversation summary
    print("\n📊 Conversation Summary:")
    print(f"  Messages: {len(conversation)}")
    print(f"  Audio files: {len(audio_files)}")
    print(f"  Export size: {len(zip_data)} bytes")
    
    return conversation, audio_files, zip_data


def main():
    """Run the demo."""
    print("🚀 Starting Streamlit TTS Chatbot Demo")
    print("This demo shows the core functionality without external dependencies.\n")
    
    # Run demo
    conversation, audio_files, zip_data = asyncio.run(demo_conversation())
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the full application:")
    print("1. Install dependencies: python3 setup.py")
    print("2. Configure secrets: cp .streamlit/secrets.toml.example .streamlit/secrets.toml")
    print("3. Run app: streamlit run app.py")


if __name__ == "__main__":
    main()
