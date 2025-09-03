"""
Streamlit Chatbot with Realtime TTS using ElevenLabs.
"""
import streamlit as st
import asyncio
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from fractions import Fraction

# Set up logging
logger = logging.getLogger(__name__)

# Import our modules
from llm.streaming_llm import create_streaming_llm
from realtime.tts_elevenlabs_ws import create_tts_client
from realtime.audio_sender import get_track, AudioQueueManager
from utils.audio import write_wav
from utils.zipper import build_conversation_zip
from stt.stt_service import create_stt_service
from stt.audio_receiver import AudioReceiverTrack

# WebRTC imports
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from aiortc.mediastreams import AudioStreamTrack
import av
import time
import logging
import threading, asyncio
from streamlit.runtime.scriptrunner import add_script_run_ctx


RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
audio_chunk_queue = Queue()

# Page configuration
st.set_page_config(
    page_title="Realtime TTS",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
        text-align: center;
    }
    .status-connected { background-color: #d4edda; color: #155724; }
    .status-connecting { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_audio_chunks" not in st.session_state:
        st.session_state.current_audio_chunks = []

    if "connection_status" not in st.session_state:
        st.session_state.connection_status = "disconnected"
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    # STT related state
    if "stt_service" not in st.session_state:
        st.session_state.stt_service = None
    
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""
    if "transcription_timeout" not in st.session_state:
        st.session_state.transcription_timeout = False


def get_config_from_secrets():
    """Get configuration from Streamlit secrets or environment variables."""
    config = {
        "elevenlabs_api_key": st.secrets.get("ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY", "")),
        "elevenlabs_voice_id": st.secrets.get("ELEVENLABS_VOICE_ID", os.getenv("ELEVENLABS_VOICE_ID", "default_voice")),
        "openai_api_key": st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),

        "available_voices": st.secrets.get("AVAILABLE_VOICES", "default_voice").split(","),
        "tts_sample_rate": int(st.secrets.get("TTS_SAMPLE_RATE", "16000")),
        "tts_chunk_size_ms": int(st.secrets.get("TTS_CHUNK_SIZE_MS", "20"))
    }
    return config


def render_header():
    """Render the main header with status."""
    st.markdown('<h1 class="main-header">🎤 Realtime TTS Tool</h1>', unsafe_allow_html=True)
    
    # Status badge
    status_class = {
        "connected": "status-connected",
        "connecting": "status-connecting", 
        "error": "status-error",
        "disconnected": "status-error"
    }.get(st.session_state.connection_status, "status-error")
    
    status_text = {
        "connected": "Connected to ElevenLabs WS",
        "connecting": "Connecting...",
        "error": "Connection Error",
        "disconnected": "Disconnected"
    }.get(st.session_state.connection_status, "Unknown")
    
    st.markdown(f'<div class="status-badge {status_class}">{status_text}</div>', unsafe_allow_html=True)


def render_sidebar(config: Dict[str, Any]):
    """Render the sidebar with controls."""
    st.sidebar.title("🎛️ Controls")
    
    # Voice selection
    selected_voice = st.sidebar.selectbox(
        "Voice",
        options=config["available_voices"],
        index=0 if config["elevenlabs_voice_id"] in config["available_voices"] else 0,
        key="voice_selectbox"
    )
    

    
    # Save conversation button
    if st.sidebar.button("💾 Save Conversation", type="primary"):
        if st.session_state.messages:
            save_conversation()
        else:
            st.sidebar.warning("No conversation to save!")
    
    # Connection info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Connection Info**")
    st.sidebar.markdown(f"Sample Rate: {config['tts_sample_rate']} Hz")
    st.sidebar.markdown(f"Chunk Size: {config['tts_chunk_size_ms']} ms")
    
    return {
        "selected_voice": selected_voice
    }


def render_chat_history():
    """Render the chat message history."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["text"])
            if message.get("audio_path"):
                st.audio(message["audio_path"], format="audio/wav")
    
    st.markdown('</div>', unsafe_allow_html=True)


async def process_user_message(user_input: str, sidebar_config: Dict[str, Any], config: Dict[str, Any]):
    """Process user message and generate streaming response with synchronized text + audio."""
    # Ensure session state is initialized
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add user message to history
    user_message = {
        "role": "user",
        "text": user_input,
        "started_at": datetime.now().isoformat(),
        "ended_at": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_message)
    st.session_state.current_audio_chunks = []

    # Placeholder for streaming text
    text_placeholder = st.empty()
    current_text_ref = [""]

    try:
        # Validate API keys
        if not config["elevenlabs_api_key"]:
            st.error("❌ ElevenLabs API key is required. Please configure it in your secrets.")
            return
        
        if not config["openai_api_key"]:
            st.error("❌ OpenAI API key is required. Please configure it in your secrets.")
            return
        
        # Create fresh clients inside this loop
        tts_client = create_tts_client(
            voice_id=sidebar_config["selected_voice"],
            api_key=config["elevenlabs_api_key"],
            sample_rate=config["tts_sample_rate"],
        )
        llm = create_streaming_llm(
            openai_api_key=config["openai_api_key"],
        )

        # Connect TTS
        st.session_state.connection_status = "connecting"
        ok = await tts_client.connect()
        if not ok:
            st.session_state.connection_status = "error"
            st.error("Failed to connect to TTS service.")
            return
        st.session_state.connection_status = "connected"

        async def wait_for_player_ready(timeout_s=5):
            import time, asyncio
            start = time.time()
            while time.time() - start < timeout_s:
                ctx = st.session_state.get("webrtc_ctx")
                if ctx:
                    # Check if the WebRTC context is initialized
                    if hasattr(ctx, 'state') and ctx.state is not None:
                        # If it's playing, great! If not, that's also okay - it will start when audio arrives
                        return True
                    # Also check if peer connection exists (even if not connected yet)
                    pc = getattr(ctx, "peer_connection", None)
                    if pc is not None:
                        return True
                await asyncio.sleep(0.1)
            return False

        # after TTS connect succeeded
        ready = await wait_for_player_ready()
        if not ready:
            st.info("WebRTC player initializing... audio will start when ready.")
        
        # Give the WebRTC player a moment to fully initialize
        await asyncio.sleep(0.5)

        # Audio collection task
        async def collect_audio():
            try:
                chunk_count = 0
                total_bytes = 0
                async for chunk in tts_client.audio_chunks():
                    if chunk and "logged_fmt" not in st.session_state:
                        st.session_state["logged_fmt"] = True
                        st.write(f"First 8 bytes: {chunk[:8]!r}, len={len(chunk)}")

                    st.session_state.current_audio_chunks.append(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    # NEW: feed the WebRTC player
                    audio_chunk_queue.put(chunk)
                
                print(f"Audio collection completed: {chunk_count} chunks, {total_bytes} total bytes")
            except Exception as e:
                print(f"Error collecting audio: {e}")

        collector_task = asyncio.create_task(collect_audio())

        # Stream text from LLM and send to TTS
        try:
            async for token in llm.stream_assistant_reply(user_input):
                current_text_ref[0] += token
                text_placeholder.markdown(f"**Assistant:** {current_text_ref[0]}")
                await tts_client.send_text_fragment(token)

            await tts_client.finalize()
        finally:
            await collector_task
            if hasattr(tts_client, "close"):
                await tts_client.close()

        # Save final assistant message + audio
        current_text = current_text_ref[0]
        audio_path = ""
        
        if st.session_state.current_audio_chunks:
            audio_path = save_audio_chunks(st.session_state.current_audio_chunks, config)
        else:
            print("No audio chunks to save!")

        assistant_message = {
            "role": "assistant",
            "text": current_text,
            "audio_path": audio_path,
            "started_at": datetime.now().isoformat(),
            "ended_at": datetime.now().isoformat(),
        }
        st.session_state.messages.append(assistant_message)

        text_placeholder.markdown(f"**Assistant:** {current_text}")
        st.session_state.current_audio_chunks = []

    except Exception as e:
        st.error(f"Error processing message: {e}")
        # Clean up TTS client on error
        try:
            if 'tts_client' in locals() and hasattr(tts_client, "close"):
                await tts_client.close()
        except:
            pass


def save_audio_chunks(chunks: List[bytes], config: Dict[str, Any]) -> str:
    """Save audio chunks to WAV file."""
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assistant_{timestamp}.wav"
        filepath = os.path.join(temp_dir, filename)
        
        # Write WAV file
        write_wav(chunks, filepath, sample_rate=config["tts_sample_rate"])
        
        # Verify file was created
        if os.path.exists(filepath):
            return filepath
        else:
            print("Error: Audio file was not created!")
            return ""
        
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return ""


def save_conversation():
    """Save conversation as ZIP file."""
    try:
        if not st.session_state.messages:
            st.warning("No conversation to save!")
            return
        
        # Build ZIP
        zip_bytes, filename = build_conversation_zip(st.session_state.messages)
        
        # Create download button
        st.sidebar.download_button(
            label=f"📥 Download {filename}",
            data=zip_bytes,
            file_name=filename,
            mime="application/zip"
        )
        
        st.sidebar.success(f"Conversation saved as {filename}")
        
    except Exception as e:
        st.error(f"Error saving conversation: {e}")


class AudioFrameGenerator(AudioStreamTrack):
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20):
        super().__init__()
        self.audio_chunk_queue = audio_chunk_queue
        self.sample_rate = sample_rate
        self.bytes_per_sample = 2  # s16
        self.frame_bytes = int(self.sample_rate * self.bytes_per_sample * frame_ms / 1000)
        self.last_chunk_time = time.time()
        self.current_audio = b""
        self.first_chunk_received = False
        # NEW: Add timing for proper audio pacing
        self.time_base = Fraction(1, self.sample_rate)
        self._pts = 0

    async def recv(self) -> av.AudioFrame:
        import logging, asyncio
        logger = logging.getLogger(__name__)

        # Fill buffer up to one frame
        while len(self.current_audio) < self.frame_bytes:
            try:
                chunk = self.audio_chunk_queue.get(timeout=0.1)
                if not self.first_chunk_received:
                    logger.info("First audio chunk received by player.")
                    self.first_chunk_received = True
                self.current_audio += chunk
            except Empty:
                await asyncio.sleep(0.01)

        frame_data = self.current_audio[:self.frame_bytes]
        self.current_audio = self.current_audio[self.frame_bytes:]

        samples = len(frame_data) // self.bytes_per_sample
        frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(frame_data)

        # NEW: timestamp & pacing
        frame.pts = self._pts
        frame.time_base = self.time_base
        self._pts += samples
        await asyncio.sleep(samples / self.sample_rate)

        return frame


def render_webrtc_player():
    if "webrtc_ctx" not in st.session_state:
        st.session_state.webrtc_ctx = None

    # Only render if we have an audio frame generator
    if "audio_frame_generator" in st.session_state:
        webrtc_ctx = webrtc_streamer(
            key="speech",
            mode=WebRtcMode.SENDONLY,
            source_audio_track=st.session_state.audio_frame_generator,
            media_stream_constraints={"video": False, "audio": True},
            frontend_rtc_configuration=RTC_CFG,
            server_rtc_configuration=RTC_CFG,
            async_processing=True,
        )
        st.session_state.webrtc_ctx = webrtc_ctx
    else:
        st.info("Initializing audio player...")


def render_stt_interface():
    """Render Speech-to-Text interface."""
    st.subheader("🎤 Voice Input")
    
    # Initialize STT service if not already done
    if st.session_state.stt_service is None:
        try:
            st.session_state.stt_service = create_stt_service(
                api_key=st.session_state.config["openai_api_key"]
            )
            
            # Set up transcription callback
            def on_transcription(text):
                if text and text.strip():
                    st.session_state.transcribed_text = text.strip()
                    # Clear transcription timeout
                    if hasattr(st.session_state, 'transcription_start_time'):
                        del st.session_state.transcription_start_time
                    st.success(f"🎉 Transcription received: {text.strip()}")
                    st.rerun()
                else:
                    st.warning("⚠️ Empty transcription received")
            
            def on_error(error):
                st.error(f"STT Error: {error}")
            
            st.session_state.stt_service.set_transcription_callback(on_transcription)
            st.session_state.stt_service.set_error_callback(on_error)
            
            st.success("STT service initialized!")
        except Exception as e:
            st.error(f"Failed to initialize STT service: {e}")
            return
    
    # Connection status
    if st.session_state.stt_service:
        stats = st.session_state.stt_service.get_recording_stats()
        if stats["is_connected"]:
            st.success("✅ Connected to OpenAI Realtime API")
        else:
            st.error("❌ Not connected to OpenAI Realtime API")
    
    # WebRTC microphone capture
    st.write("**Microphone Access:**")
    
    def audio_frame_callback(frame):
        """Callback for processing audio frames from microphone."""
        if st.session_state.stt_service and st.session_state.is_recording:
            # Convert frame to bytes and process
            audio_data = frame.to_ndarray()
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=0)  # Convert to mono
            audio_bytes = (audio_data * 32767).astype('int16').tobytes()
            
            # Debug: Log audio frame info
            logger.info(f"Processing audio frame: {len(audio_bytes)} bytes, shape: {audio_data.shape}")
            
            # Process audio data asynchronously
            import asyncio
            try:
                asyncio.run(st.session_state.stt_service.process_audio_data(audio_bytes))
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                st.error(f"Error processing audio: {e}")
    
    # WebRTC receiver for microphone input
    webrtc_ctx = webrtc_streamer(
        key="microphone",
        mode=WebRtcMode.RECVONLY,
        media_stream_constraints={"video": False, "audio": True},
        frontend_rtc_configuration=RTC_CFG,
        server_rtc_configuration=RTC_CFG,
        async_processing=True,
    )
    
    # Store the WebRTC context in session state for debugging
    if webrtc_ctx:
        st.session_state.webrtc_mic_ctx = webrtc_ctx
    
    # Debug: Show WebRTC status
    if webrtc_ctx and hasattr(webrtc_ctx, 'audio_receiver') and webrtc_ctx.audio_receiver:
        st.success("✅ Microphone access granted")
        webrtc_ctx.audio_receiver.add_track(audio_frame_callback)
    elif webrtc_ctx and hasattr(webrtc_ctx, 'state'):
        # Check if WebRTC is in a connected state
        if webrtc_ctx.state == "PLAYING" or webrtc_ctx.state == "CONNECTED":
            st.success("✅ Microphone access granted")
            if hasattr(webrtc_ctx, 'audio_receiver') and webrtc_ctx.audio_receiver:
                webrtc_ctx.audio_receiver.add_track(audio_frame_callback)
        else:
            st.info("🔄 WebRTC initializing... Please click 'Start' to enable microphone access")
    else:
        st.info("🔄 WebRTC initializing... Please click 'Start' to enable microphone access")
    
    # Connection controls
    if st.session_state.stt_service:
        stats = st.session_state.stt_service.get_recording_stats()
        col_conn1, col_conn2 = st.columns([1, 1])
        
        with col_conn1:
            if not stats["is_connected"]:
                if st.button("🔌 Connect to OpenAI"):
                    with st.spinner("Connecting..."):
                        import asyncio
                        try:
                            asyncio.run(st.session_state.stt_service.connect())
                            st.success("Connected successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Connection failed: {e}")
        
        with col_conn2:
            if stats["is_connected"]:
                if st.button("🔌 Disconnect"):
                    with st.spinner("Disconnecting..."):
                        import asyncio
                        try:
                            asyncio.run(st.session_state.stt_service.disconnect())
                            st.success("Disconnected successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Disconnection failed: {e}")
    
    # Recording controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        stats = st.session_state.stt_service.get_recording_stats() if st.session_state.stt_service else {"is_connected": False}
        if st.button("🎤 Start Recording", disabled=st.session_state.is_recording or not stats["is_connected"]):
            logger.info("Start recording button clicked")
            st.session_state.stt_service.start_recording()
            st.session_state.is_recording = True
            st.info("🎤 Recording started - speak now!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording):
            logger.info("Stop recording button clicked")
            st.session_state.stt_service.stop_recording()
            st.session_state.is_recording = False
            st.session_state.transcription_timeout = False
            
            # Show audio buffer size for debugging
            stats = st.session_state.stt_service.get_recording_stats()
            st.info(f"📊 Audio buffer size: {stats.get('recorded_audio_size', 0)} bytes")
            
            # Transcribe the recorded audio
            with st.spinner("🔄 Processing audio..."):
                import asyncio
                result = asyncio.run(st.session_state.stt_service.commit_audio_for_transcription())
                if result["success"]:
                    st.info("🎯 Audio sent for transcription - waiting for results...")
                    # Set a timeout flag to show manual option if transcription doesn't come back
                    import time
                    st.session_state.transcription_start_time = time.time()
                else:
                    st.error(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
            st.rerun()
    
    with col3:
        if st.session_state.is_recording:
            st.error("🔴 Recording...")
        else:
            st.info("⏸️ Ready to record")
    
    # Display transcribed text and transcription status
    if st.session_state.transcribed_text:
        st.success("✅ Transcription Complete!")
        st.text_area("Transcribed Text:", value=st.session_state.transcribed_text, height=100)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📤 Send as Message"):
                # Send the transcribed text as a message
                if st.session_state.transcribed_text.strip():
                    run_coro_in_thread(process_user_message(
                        st.session_state.transcribed_text.strip(), 
                        st.session_state.get("sidebar_config", {}), 
                        st.session_state.get("config", {})
                    ))
                    st.session_state.transcribed_text = ""  # Clear after sending
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.transcribed_text = ""
                st.rerun()
    else:
        # Show transcription status when no text yet
        if st.session_state.is_recording:
            st.info("🎤 Recording... Speak now!")
        elif hasattr(st.session_state, 'stt_service') and st.session_state.stt_service.is_connected:
            # Check for transcription timeout
            if hasattr(st.session_state, 'transcription_start_time'):
                import time
                elapsed = time.time() - st.session_state.transcription_start_time
                if elapsed > 10:  # 10 second timeout
                    st.warning("⏰ Transcription taking longer than expected...")
                    if st.button("🔄 Retry Transcription"):
                        # Clear timeout and retry
                        del st.session_state.transcription_start_time
                        st.rerun()
                else:
                    st.info("🎯 Ready to record - click 'Start Recording' to begin")
            else:
                st.info("🎯 Ready to record - click 'Start Recording' to begin")
            
            # Debug: Add manual transcription test button
            if st.button("🧪 Test Transcription (Debug)"):
                # Simulate a transcription for testing
                st.session_state.transcribed_text = "This is a test transcription"
                if hasattr(st.session_state, 'transcription_start_time'):
                    del st.session_state.transcription_start_time
                st.rerun()
    
    # STT stats
    if st.session_state.stt_service:
        with st.expander("📊 STT Statistics"):
            stats = st.session_state.stt_service.get_recording_stats()
            st.json(stats)


def generate_and_queue_audio(llm_response_generator, tts_model, tts_settings):
    """
    Generates audio from the LLM response stream and puts it into a queue.
    This function is intended to be run in a separate thread.
    """
    logger = logging.getLogger(__name__)

    # Get the audio stream from the TTS model
    audio_chunks = tts_model.stream_tts_from_llm(
        llm_response_generator,
        **tts_settings
    )

    total_chunks = 0
    total_bytes = 0

    # Iterate through the audio chunks and put them in the queue
    for chunk in audio_chunks:
        audio_chunk_queue.put(chunk)
        total_chunks += 1
        total_bytes += len(chunk)

    logger.info(f"Audio generation completed: {total_chunks} chunks, {total_bytes} total bytes")


def on_streaming_start():
    st.session_state.streaming = True
    st.session_state.llm_response_text = ""


def on_streaming_stop():
    st.session_state.streaming = False

def fire(coro):
    """Fire a coroutine from a daemon thread."""
    threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()

def run_coro_in_thread(coro):
    t = threading.Thread(target=lambda: asyncio.run(coro), daemon=True)
    add_script_run_ctx(t)  # <-- critical
    t.start()

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Get configuration
    config = get_config_from_secrets()

    # Create audio generator with correct sample rate & frame size
    if "audio_frame_generator" not in st.session_state:
        st.session_state.audio_frame_generator = AudioFrameGenerator(
            sample_rate=config["tts_sample_rate"],
            frame_ms=config["tts_chunk_size_ms"],
        )

    # Render header
    render_header()

    # Render sidebar
    sidebar_config = render_sidebar(config)
    
    # Store config in session state for STT interface
    st.session_state.config = config
    st.session_state.sidebar_config = sidebar_config

    # Render chat history
    render_chat_history()

    st.info("Click **Start** in the audio player once, then messages will stream with voice.")

    # WebRTC audio player
    render_webrtc_player()
    
    # STT Interface
    render_stt_interface()

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # fire(process_user_message(prompt, sidebar_config, config))
        run_coro_in_thread(process_user_message(prompt, sidebar_config, config))

    # System status
    with st.expander("🔧 System Status"):
        st.json({
            "connection_status": st.session_state.connection_status,
            "message_count": len(st.session_state.messages),
            "audio_chunks_ready": len(st.session_state.current_audio_chunks) > 0,
        })
    
    # WebRTC debug
    with st.expander("🔧 WebRTC Debug"):
        # Audio player context
        ctx = st.session_state.get("webrtc_ctx")
        st.write("**Audio Player WebRTC:**")
        st.write("State:", getattr(ctx, "state", None))
        if ctx:
            st.write("Has audio_receiver:", hasattr(ctx, 'audio_receiver'))
            if hasattr(ctx, 'audio_receiver'):
                st.write("Audio receiver:", ctx.audio_receiver)
        
        # Microphone context
        mic_ctx = st.session_state.get("webrtc_mic_ctx")
        st.write("**Microphone WebRTC:**")
        st.write("State:", getattr(mic_ctx, "state", None))
        if mic_ctx:
            st.write("Has audio_receiver:", hasattr(mic_ctx, 'audio_receiver'))
            if hasattr(mic_ctx, 'audio_receiver'):
                st.write("Audio receiver:", mic_ctx.audio_receiver)
            st.write("Available attributes:", [attr for attr in dir(mic_ctx) if not attr.startswith('_')])



if __name__ == "__main__":
    main()

