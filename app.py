"""
Streamlit Chatbot with Realtime TTS using ElevenLabs.
"""
import streamlit as st
import asyncio
import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from queue import Queue, Empty
from fractions import Fraction
import toml
import json

# Set up logging (only once)
logger = logging.getLogger(__name__)
if not logger.handlers:  # Prevent duplicate handlers
    logger.setLevel(logging.WARNING)  # Reduced from DEBUG to INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
import signal
from contextlib import contextmanager
import atexit

# Frontend/browser ICE: keep STUN here
FRONTEND_RTC_CFG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302",
                  "stun:stun1.l.google.com:19302",
                  "stun:stun2.l.google.com:19302",
                  "stun:stun3.l.google.com:19302",
                  "stun:stun4.l.google.com:19302"]},
    ],
    # Keep browser-only tweaks here if you want:
    "iceCandidatePoolSize": 10,
    "iceTransportPolicy": "all",
    "bundlePolicy": "max-bundle",
    "rtcpMuxPolicy": "require",
})

# Server/aiortc ICE: Use STUN for proper connection establishment
SERVER_RTC_CFG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ],
    "iceCandidatePoolSize": 10,
    "iceTransportPolicy": "all",
    "bundlePolicy": "max-bundle",
    "rtcpMuxPolicy": "require",
})

audio_chunk_queue = Queue()

# Global thread-safe queue for microphone audio data
# This is accessible from any thread, unlike st.session_state
mic_pcm_q_global = Queue()
ui_event_q = Queue()      # -> for UI notifications from worker threads
raw_audio_q = Queue()     # -> float32 mono 48kHz chunks for optional recording

APP_SHUTTING_DOWN = threading.Event()

def cleanup_on_exit():
    # Do NOT call st.* or read session_state here.
    APP_SHUTTING_DOWN.set()

atexit.register(cleanup_on_exit)

# Audio recording state
if "recorded_audio_data" not in st.session_state:
    st.session_state.recorded_audio_data = []
if "is_recording_audio" not in st.session_state:
    st.session_state.is_recording_audio = False

def on_audio_frames(frames):
    """Process mic frames in worker thread; never call st.* here."""
    if not frames:
        logger.warning("on_audio_frames: empty frames")
        return

    for i, frame in enumerate(frames):
        try:
            arr = frame.to_ndarray()
            if arr.ndim == 2:  # stereo -> mono
                arr = arr.mean(axis=0)
            if arr.dtype != np.float32:  # to float32 in [-1,1]
                arr = (arr.astype(np.float32) / 32767.0)

            # For optional file recording on the main thread:
            try:
                raw_audio_q.put_nowait(arr.copy())  # 48kHz mono float32
            except Exception:
                pass

            # Resample to 16kHz for STT
            try:
                from scipy.signal import resample_poly
                arr16k = resample_poly(arr, up=1, down=3)  # 48k -> 16k
            except Exception as e:
                logger.debug(f"resample_poly unavailable/failed ({e}); decimating")
                arr16k = arr[::3]

            pcm16 = (np.clip(arr16k, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

            # Feed global queue for STT consumer (and anything else)
            mic_pcm_q_global.put_nowait(pcm16)
        except Exception:
            logger.exception(f"on_audio_frames: error processing frame {i}")

def save_recorded_audio():
    """Save recorded audio data to a WAV file."""
    if not st.session_state.recorded_audio_data:
        return None
    
    try:
        import soundfile as sf
        from datetime import datetime
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(st.session_state.recorded_audio_data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_recording_{timestamp}.wav"
        
        # Save as WAV file (48kHz, mono)
        sf.write(filename, audio_data, 48000)
        
        # Clear recorded data
        st.session_state.recorded_audio_data = []
        
        return filename
    except ImportError:
        st.error("soundfile library not available. Install with: pip install soundfile")
        return None
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return None

def safe_webrtc_streamer(*args, **kwargs):
    """Safely create a WebRTC streamer with error handling."""
    try:
        # Add timeout and better error handling
        kwargs.setdefault('frontend_rtc_configuration', FRONTEND_RTC_CFG)
        kwargs.setdefault('server_rtc_configuration', SERVER_RTC_CFG)
        kwargs.setdefault('media_stream_constraints', {"video": False, "audio": True})
        kwargs.setdefault('async_processing', True)
        
        return webrtc_streamer(*args, **kwargs)
    except Exception as e:
        logger.error(f"WebRTC initialization failed: {e}")
        # Log more details about the error
        import traceback
        logger.error(f"WebRTC error traceback: {traceback.format_exc()}")
        return None

def cleanup_webrtc_connections():
    """Best-effort cleanup that doesn't assume .stop() exists."""
    for key in ("webrtc_ctx", "webrtc_mic_ctx"):
        ctx = st.session_state.get(key)
        if not ctx:
            continue
        try:
            # 1) Preferred public-ish methods if present
            if hasattr(ctx, "destroy"):
                try:
                    ctx.destroy()
                except Exception:
                    pass
            elif hasattr(ctx, "close"):
                try:
                    ctx.close()
                except Exception:
                    pass

            # 2) Close the underlying PeerConnection if exposed
            pc = getattr(ctx, "peer_connection", None) or getattr(ctx, "pc", None)
            if pc:
                try:
                    pc.close()
                except Exception:
                    pass

            # 3) Stop receivers if they expose stop()
            ar = getattr(ctx, "audio_receiver", None)
            if hasattr(ar, "stop"):
                try:
                    ar.stop()
                except Exception:
                    pass

            vr = getattr(ctx, "video_receiver", None)
            if hasattr(vr, "stop"):
                try:
                    vr.stop()
                except Exception:
                    pass
        finally:
            st.session_state[key] = None


def render_webrtc_fallback():
    """Render a fallback interface when WebRTC fails."""
    st.warning("⚠️ WebRTC connection failed. Using fallback mode.")
    st.info("You can still use text input, but voice features are limited.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retry WebRTC Connection"):
            cleanup_webrtc_connections()
            st.rerun()
    with col2:
        if st.button("🧹 Cleanup & Retry"):
            cleanup_webrtc_connections()
            # Force a complete page refresh
            st.experimental_rerun()

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
    
    # Audio processing queue - reference to global thread-safe queue
    if "mic_pcm_q" not in st.session_state:
        st.session_state.mic_pcm_q = mic_pcm_q_global
    
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""
    if "transcription_timeout" not in st.session_state:
        st.session_state.transcription_timeout = False


def load_config_from_file(file_content: str, file_type: str) -> Dict[str, Any]:
    """Load configuration from uploaded file content."""
    try:
        if file_type == "toml":
            raw_config = toml.loads(file_content)
        elif file_type == "json":
            raw_config = json.loads(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Flatten nested config structure
        flattened_config = {}
        
        # Handle nested structures like [api_keys] and [tts_settings]
        for key, value in raw_config.items():
            if isinstance(value, dict):
                # If it's a nested dict, flatten it
                for nested_key, nested_value in value.items():
                    flattened_config[nested_key] = nested_value
            else:
                # Direct key-value pair
                flattened_config[key] = value
        
        return flattened_config
        
    except Exception as e:
        st.error(f"Error parsing config file: {str(e)}")
        return {}


def get_config_from_secrets():
    """Get configuration with fallback priority: secrets.toml -> uploaded config -> environment variables."""
    # Default config structure
    default_config = {
        "elevenlabs_api_key": "",
        "elevenlabs_voice_id": "default_voice",
        "openai_api_key": "",
        "available_voices": ["default_voice"],
        "tts_sample_rate": 16000,
        "tts_chunk_size_ms": 20
    }
    
    # Priority 1: Streamlit secrets (secrets.toml) - handle missing secrets gracefully
    try:
        # Check if secrets.toml exists by trying to access it
        _ = st.secrets.get("ELEVENLABS_API_KEY", "")
        # If we get here, secrets.toml exists, so use it
        config = {
            "elevenlabs_api_key": st.secrets.get("ELEVENLABS_API_KEY", ""),
            "elevenlabs_voice_id": st.secrets.get("ELEVENLABS_VOICE_ID", "default_voice"),
            "openai_api_key": st.secrets.get("OPENAI_API_KEY", ""),
            "available_voices": st.secrets.get("AVAILABLE_VOICES", "default_voice").split(","),
            "tts_sample_rate": int(st.secrets.get("TTS_SAMPLE_RATE", "16000")),
            "tts_chunk_size_ms": int(st.secrets.get("TTS_CHUNK_SIZE_MS", "20"))
        }
    except Exception:
        # If secrets.toml is missing or invalid, start with default config
        config = default_config.copy()
    
    # Priority 2: Uploaded config file (if available)
    if "uploaded_config" in st.session_state and st.session_state.uploaded_config:
        uploaded_config = st.session_state.uploaded_config
        # Merge uploaded config, only override if values are not empty
        for key, value in uploaded_config.items():
            if value and value != "" and value != []:
                config[key] = value
    
    # Priority 3: Environment variables (fallback)
    for key in ["elevenlabs_api_key", "openai_api_key"]:
        if not config[key]:
            config[key] = os.getenv(key.upper(), "")
    
    # Ensure required fields have defaults
    for key, default_value in default_config.items():
        if key not in config or not config[key]:
            config[key] = default_value
    
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
    
    # Configuration Upload Section
    st.sidebar.markdown("### 🔧 Configuration")
    
    # Initialize uploaded config in session state
    if "uploaded_config" not in st.session_state:
        st.session_state.uploaded_config = None
    
    # File uploader for config
    uploaded_file = st.sidebar.file_uploader(
        "Upload Config File",
        type=['toml', 'json'],
        help="Upload a TOML or JSON config file with your API keys. This will override secrets.toml if present.",
        key="config_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read file content
            file_content = uploaded_file.read().decode('utf-8')
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Parse config
            parsed_config = load_config_from_file(file_content, file_type)
            
            if parsed_config:
                st.session_state.uploaded_config = parsed_config
                # Update the main config in session state with the new uploaded config
                st.session_state.config = get_config_from_secrets()
                st.sidebar.success(f"✅ Config loaded from {uploaded_file.name}")
                
                # Show loaded config summary
                with st.sidebar.expander("📋 Loaded Configuration", expanded=False):
                    for key, value in parsed_config.items():
                        if "api_key" in key.lower():
                            # Mask API keys for security
                            masked_value = f"{value[:8]}..." if value and len(value) > 8 else "Not set"
                            st.write(f"**{key}**: {masked_value}")
                        else:
                            st.write(f"**{key}**: {value}")
            else:
                st.sidebar.error("❌ Failed to parse config file")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading config: {str(e)}")
    
    # Clear config button
    if st.session_state.uploaded_config:
        if st.sidebar.button("🗑️ Clear Uploaded Config", type="secondary"):
            st.session_state.uploaded_config = None
            # Update the main config in session state after clearing uploaded config
            st.session_state.config = get_config_from_secrets()
            st.sidebar.success("Config cleared")
            st.rerun()
    
    # Show config source status
    try:
        # Check if secrets.toml exists by trying to access it
        _ = st.secrets.get("ELEVENLABS_API_KEY", "")
        has_secrets = True
    except Exception:
        has_secrets = False
    
    if has_secrets and not st.session_state.uploaded_config:
        config_source = "secrets.toml"
        st.sidebar.info(f"📁 Using config from: {config_source}")
    elif st.session_state.uploaded_config:
        config_source = "uploaded file"
        st.sidebar.info(f"📁 Using config from: {config_source}")
    else:
        st.sidebar.warning("⚠️ No config found! Please upload a config file below.")
    
    # Sample config download
    sample_config_toml = """# Sample Configuration File
# Copy this file and fill in your API keys

[api_keys]
elevenlabs_api_key = "your_elevenlabs_api_key_here"
openai_api_key = "your_openai_api_key_here"

[tts_settings]
elevenlabs_voice_id = "default_voice"
available_voices = ["voice1", "voice2", "voice3"]
tts_sample_rate = 16000
tts_chunk_size_ms = 20
"""
    
    st.sidebar.download_button(
        "📥 Download Sample Config",
        sample_config_toml,
        file_name="sample_config.toml",
        mime="text/plain",
        help="Download a sample TOML config file template"
    )
    
    st.sidebar.markdown("---")
    
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
    # Use ui_event_q to handle session state updates from main thread
    user_message = {
        "role": "user",
        "text": user_input,
        "started_at": datetime.now().isoformat(),
        "ended_at": datetime.now().isoformat()
    }
    ui_event_q.put(("add_user_message", user_message))
    ui_event_q.put(("clear_audio_chunks", None))

    # Placeholder for streaming text - use ui_event_q to get placeholder
    ui_event_q.put(("get_text_placeholder", None))
    current_text_ref = [""]

    try:
        # Validate API keys
        if not config["elevenlabs_api_key"]:
            ui_event_q.put(("error", "❌ ElevenLabs API key is required. Please configure it in your secrets."))
            return
        
        if not config["openai_api_key"]:
            ui_event_q.put(("error", "❌ OpenAI API key is required. Please configure it in your secrets."))
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
        ui_event_q.put(("connection_status", "connecting"))
        ok = await tts_client.connect()
        if not ok:
            ui_event_q.put(("connection_status", "error"))
            ui_event_q.put(("error", "Failed to connect to TTS service."))
            return
        ui_event_q.put(("connection_status", "connected"))

        async def wait_for_player_ready(timeout_s=5):
            import time, asyncio
            start = time.time()
            while time.time() - start < timeout_s:
                # Use ui_event_q to request WebRTC context from main thread
                ui_event_q.put(("webrtc_status_request", None))
                await asyncio.sleep(0.1)
                # For now, just return True after a short delay
                # The main thread will handle WebRTC status checking
                return True
            return False

        # after TTS connect succeeded
        ready = await wait_for_player_ready()
        if not ready:
            # Use ui_event_q to notify main thread about WebRTC status
            ui_event_q.put(("webrtc_info", "WebRTC player initializing... audio will start when ready."))
        
        # Give the WebRTC player a moment to fully initialize
        await asyncio.sleep(0.5)

        # Audio collection task
        collected_chunks = []
        async def collect_audio():
            try:
                chunk_count = 0
                total_bytes = 0
                logged_fmt = False
                async for chunk in tts_client.audio_chunks():
                    if chunk and not logged_fmt:
                        logged_fmt = True
                        logger.debug(f"Audio chunk len={len(chunk)}")

                    # Use ui_event_q to notify main thread about audio chunks
                    ui_event_q.put(("audio_chunk", chunk))
                    chunk_count += 1
                    total_bytes += len(chunk)
                    collected_chunks.append(chunk)
                    # NEW: feed the WebRTC player
                    audio_chunk_queue.put(chunk)
                
                logger.info(f"Audio collection completed: {chunk_count} chunks, {total_bytes} total bytes")
            except Exception as e:
                logger.error(f"Error collecting audio: {e}")

        collector_task = asyncio.create_task(collect_audio())

        # Stream text from LLM and send to TTS
        try:
            async for token in llm.stream_assistant_reply(user_input):
                current_text_ref[0] += token
                ui_event_q.put(("update_text_placeholder", f"**Assistant:** {current_text_ref[0]}"))
                await tts_client.send_text_fragment(token)

            await tts_client.finalize()
        finally:
            await collector_task
            if hasattr(tts_client, "close"):
                await tts_client.close()

        # Save final assistant message + audio
        current_text = current_text_ref[0]

        # 🚫 remove any direct session_state writes here
        # ✅ enqueue a single event with everything the UI needs:
        ui_event_q.put((
            "finalize_assistant",
            {
                "text": current_text,
                "chunks": collected_chunks,   # list[bytes]
                "config": config,             # for sample rate, etc.
            }
        ))

    except Exception as e:
        ui_event_q.put(("error", f"Error processing message: {e}"))
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
        try:
            webrtc_ctx = safe_webrtc_streamer(
                key="player",
                mode=WebRtcMode.RECVONLY,
                source_audio_track=st.session_state.audio_frame_generator,  # server → browser
                media_stream_constraints={"video": False, "audio": False},
                frontend_rtc_configuration=FRONTEND_RTC_CFG,
                server_rtc_configuration=SERVER_RTC_CFG,
                async_processing=True,
                sendback_audio=False,                          # avoid echo
            )
        except Exception as e:
            logger.error(f"Failed to create audio player WebRTC streamer: {e}")
            import traceback
            logger.error(f"WebRTC error traceback: {traceback.format_exc()}")
            webrtc_ctx = None
        if webrtc_ctx is None:
            st.error("Failed to initialize audio player. Please refresh the page and try again.")
            st.info("If the problem persists, try using a different browser or check your internet connection.")
        st.session_state.webrtc_ctx = webrtc_ctx
    else:
        st.info("Initializing audio player...")

def _audio_consumer_loop():
    while not APP_SHUTTING_DOWN.is_set():
        try:
            pcm16 = mic_pcm_q_global.get(timeout=0.2)
        except Empty:
            continue
        try:
            # Safe read with add_script_run_ctx, but skip if shutting down
            if APP_SHUTTING_DOWN.is_set():
                break
            stt_service = st.session_state.get("stt_service")
            is_rec = st.session_state.get("is_recording", False)
            if stt_service and is_rec:
                asyncio.run(stt_service.process_audio_data(pcm16))
        except Exception:
            logger.exception("audio_consumer_loop error")

def render_stt_interface():
    """Render Speech-to-Text interface."""
    st.subheader("🎤 Voice Input")
    
    # For now, skip STT service initialization and go straight to WebRTC testing
    st.info("🧪 WebRTC Microphone Test Mode - STT service disabled for debugging")

    def _pump_ui_events():
        pumped = 0
        while True:
            try:
                kind, payload = ui_event_q.get_nowait()
            except Empty:
                break
            if kind == "transcription":
                if payload:
                    st.session_state.transcribed_text = payload
                    st.session_state._needs_rerun = True
            elif kind == "error":
                # safe to call st.* here (we are on main thread)
                st.error(payload)
            elif kind == "audio_chunk":
                # Handle audio chunks from collect_audio() - add to session state
                st.session_state.current_audio_chunks.append(payload)
            elif kind == "add_user_message":
                # Add user message to session state
                st.session_state.messages.append(payload)
            elif kind == "add_assistant_message":
                # Add assistant message to session state
                st.session_state.messages.append(payload)
            elif kind == "clear_audio_chunks":
                # Clear audio chunks
                st.session_state.current_audio_chunks = []
            elif kind == "connection_status":
                # Update connection status
                st.session_state.connection_status = payload
            elif kind == "webrtc_info":
                # Show WebRTC info message
                st.info(payload)
            elif kind == "webrtc_status_request":
                # Handle WebRTC status request (placeholder for now)
                pass
            elif kind == "get_text_placeholder":
                # Get text placeholder - create one if it doesn't exist
                if "text_placeholder" not in st.session_state:
                    st.session_state.text_placeholder = st.empty()
            elif kind == "update_text_placeholder":
                # Update text placeholder
                if "text_placeholder" in st.session_state:
                    st.session_state.text_placeholder.markdown(payload)
                # 🔁 request a rerun so the UI updates during streaming
                # (optional throttle to avoid too many reruns)
                cnt = st.session_state.get("_stream_tick", 0) + 1
                st.session_state["_stream_tick"] = cnt
                if cnt % 3 == 0:  # update UI every 3 tokens (tweak as you like)
                    st.session_state._needs_rerun = True
            elif kind == "save_audio_chunks":
                # Save audio chunks
                if st.session_state.current_audio_chunks:
                    audio_path = save_audio_chunks(st.session_state.current_audio_chunks, payload)
                    st.session_state.current_audio_path = audio_path
            elif kind == "finalize_assistant":
                payload = payload or {}
                text = payload.get("text", "")
                chunks = payload.get("chunks") or []
                cfg = payload.get("config") or st.session_state.get("config", {})
                audio_path = ""
                if chunks:
                    audio_path = save_audio_chunks(chunks, cfg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": text,
                    "audio_path": audio_path,
                    "started_at": datetime.now().isoformat(),
                    "ended_at": datetime.now().isoformat(),
                })
                # update the live placeholder too if present
                if "text_placeholder" in st.session_state:
                    st.session_state.text_placeholder.markdown(f"**Assistant:** {text}")
            elif kind == "get_audio_path":
                # Get audio path (placeholder for now)
                pass
            pumped += 1
        return pumped
    
    _pump_ui_events()

    # Initialize STT service if not already done or if config changed
    current_api_key = st.session_state.config.get("openai_api_key", "")
    if (st.session_state.stt_service is None or 
        not hasattr(st.session_state, 'last_openai_api_key') or 
        st.session_state.last_openai_api_key != current_api_key):
        
        # Clean up existing service if it exists
        if st.session_state.stt_service:
            try:
                st.session_state.stt_service.close()
            except:
                pass
            st.session_state.stt_service = None
        
        try:
            # Try to import and create STT service
            try:
                st.session_state.stt_service = create_stt_service(
                    api_key=current_api_key
                )
                st.session_state.last_openai_api_key = current_api_key
                
                # Set up transcription callback
                def on_transcription(text):
                    # Worker thread -> just enqueue event; main thread will render
                    ui_event_q.put(("transcription", (text or "").strip()))

                def on_error(error):
                    ui_event_q.put(("error", str(error)))
                
                st.session_state.stt_service.set_transcription_callback(on_transcription)
                st.session_state.stt_service.set_error_callback(on_error)
                
                st.success("STT service initialized!")
            except ImportError as e:
                logger.error(f"STT service import failed: {e}")
                st.warning(f"⚠️ STT service import failed: {e} - continuing in test mode")
                st.session_state.stt_service = None
            except Exception as e:
                logger.error(f"STT service creation failed: {e}")
                st.warning(f"⚠️ STT service creation failed: {e} - continuing in test mode")
                st.session_state.stt_service = None
        except Exception as e:
            st.error(f"Failed to initialize STT service: {e}")
            st.warning("⚠️ WebRTC microphone will still work for testing, but STT features are disabled")
    
    # Connection status
    if st.session_state.stt_service:
        stats = st.session_state.stt_service.get_recording_stats()
        if stats["is_connected"]:
            st.success("✅ Connected to OpenAI Realtime API")
        else:
            st.error("❌ Not connected to OpenAI Realtime API")
    else:
        st.warning("⚠️ STT service not available - WebRTC microphone is in test mode")
    
    # WebRTC microphone capture
    st.write("**Microphone Access:**")
    
    # WebRTC receiver for microphone input with better error handling
    try:
        logger.info("🎤 Creating WebRTC microphone streamer...")
        logger.info(f"🔧 Frontend RTC config: {FRONTEND_RTC_CFG}")
        logger.info(f"🔧 Server RTC config: {SERVER_RTC_CFG}")
        logger.info(f"🎯 Audio callback: {on_audio_frames}")
        
        webrtc_ctx = safe_webrtc_streamer(
            key="microphone",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={
                "video": False, 
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True
                }
            },
            frontend_rtc_configuration=FRONTEND_RTC_CFG,
            server_rtc_configuration=SERVER_RTC_CFG,
            async_processing=True,
            queued_audio_frames_callback=on_audio_frames,  # mic → STT
            audio_receiver_size=8,                         # small buffer
            sendback_audio=False,                          # avoid echo
        )
        
        # Only log WebRTC creation once per session to reduce spam
        if "webrtc_created" not in st.session_state:
            logger.info(f"✅ WebRTC microphone streamer created: {webrtc_ctx}")
            st.session_state.webrtc_created = True
            if webrtc_ctx:
                logger.debug(f"📊 WebRTC context state: {getattr(webrtc_ctx, 'state', 'No state')}")
                logger.debug(f"🔗 WebRTC context attributes: {[attr for attr in dir(webrtc_ctx) if not attr.startswith('_')]}")
    except Exception as e:
        logger.error(f"Failed to create microphone WebRTC streamer: {e}")
        import traceback
        logger.error(f"WebRTC error traceback: {traceback.format_exc()}")
        webrtc_ctx = None
        
        # Show error in UI for debugging
        st.error(f"❌ WebRTC Error: {e}")
        st.code(traceback.format_exc())
    
    if webrtc_ctx is None:
        st.error("Failed to initialize microphone. Please refresh the page and try again.")
        st.info("If the problem persists, try using a different browser or check your internet connection.")
        st.info("Make sure your browser allows microphone access for this site.")
        
        # Add a test button even when WebRTC fails
        st.write("**Debug Options:**")
        if st.button("🧪 Test Callback (No WebRTC)"):
            # Create a dummy audio frame to test the callback
            import numpy as np
            from av import AudioFrame
            dummy_frame = AudioFrame.from_ndarray(
                np.random.randn(1024).astype(np.float32), 
                format='flt', 
                layout='mono'
            )
            logger.info("🧪 Testing on_audio_frames callback with dummy data (no WebRTC)")
            on_audio_frames([dummy_frame])
            st.success("✅ Callback test completed - check logs")
        
        # Show fallback interface
        render_webrtc_fallback()
        return
    
    # Store the WebRTC context in session state for debugging
    st.session_state.webrtc_mic_ctx = webrtc_ctx
    
    # Add refresh button for WebRTC issues
    col_refresh1, col_refresh2, col_refresh3, col_test = st.columns([1, 1, 1, 1])
    with col_refresh1:
        if st.button("🔄 Refresh WebRTC"):
            cleanup_webrtc_connections()
            st.rerun()
    with col_refresh2:
        if st.button("🧹 Force Cleanup"):
            cleanup_webrtc_connections()
            st.rerun()
    with col_test:
        if st.button("🧪 Test Callback"):
            # Create a dummy audio frame to test the callback
            import numpy as np
            from av import AudioFrame
            dummy_frame = AudioFrame.from_ndarray(
                np.random.randn(1024).astype(np.float32), 
                format='flt', 
                layout='mono'
            )
            logger.info("🧪 Testing on_audio_frames callback with dummy data")
            on_audio_frames([dummy_frame])
            st.success("✅ Callback test completed - check logs")
    with col_refresh3:
        st.info("💡 If WebRTC is stuck, try the cleanup buttons above")
    
    # Debug: Show WebRTC status
    if webrtc_ctx:
        try:
            # Check WebRTC state more comprehensively
            webrtc_state = getattr(webrtc_ctx, 'state', None)
            has_audio_receiver = hasattr(webrtc_ctx, 'audio_receiver') and webrtc_ctx.audio_receiver
            
            # Show current state for debugging
            st.write(f"**WebRTC State:** {webrtc_state}")
            st.write(f"**Has Audio Receiver:** {has_audio_receiver}")
            
            if webrtc_state in ["PLAYING", "CONNECTED"] or has_audio_receiver:
                st.success("✅ Microphone access granted")
            elif webrtc_state == "INITIALIZED":
                st.info("🔄 WebRTC initialized - Click 'Start' to begin recording")
            elif webrtc_state is None:
                st.info("🔄 WebRTC initializing... Please wait")
            else:
                st.info(f"🔄 WebRTC state: {webrtc_state} - Please click 'Start' to enable microphone access")
        except Exception as e:
            st.error(f"Error checking WebRTC status: {e}")
    else:
        st.error("❌ WebRTC context not available")
    
    # Connection controls
    if st.session_state.stt_service:
        stats = st.session_state.stt_service.get_recording_stats()
        col_conn1, col_conn2 = st.columns([1, 1])
        
        with col_conn1:
            if not stats["is_connected"]:
                if st.button("🔌 Connect to OpenAI"):
                    with st.spinner("Connecting..."):
                        run_coro_in_thread(st.session_state.stt_service.connect())
        
        with col_conn2:
            if stats["is_connected"]:
                if st.button("🔌 Disconnect"):
                    with st.spinner("Disconnecting..."):
                        run_coro_in_thread(st.session_state.stt_service.disconnect())
                        st.session_state._needs_rerun = True
    
    # Recording controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Audio recording status and controls
    with col3:
        if st.session_state.is_recording:
            st.error("🔴 Recording...")
            if st.session_state.is_recording_audio:
                st.info(f"🎵 Audio recording: {len(st.session_state.recorded_audio_data)} chunks")
        else:
            st.info("⏸️ Not recording")
        
        # Audio recording toggle (only show when not recording)
        if not st.session_state.is_recording:
            audio_recording_enabled = st.checkbox(
                "🎵 Save audio to file", 
                value=st.session_state.get("audio_recording_enabled", True),
                help="When enabled, your speech will be saved as a WAV file when you stop recording"
            )
            st.session_state.audio_recording_enabled = audio_recording_enabled
    
    with col1:
        stats = st.session_state.stt_service.get_recording_stats() if st.session_state.stt_service else {"is_connected": False}
        if st.button("🎤 Start Recording", disabled=st.session_state.is_recording or not stats["is_connected"]):
            logger.info("Start recording button clicked")
            st.session_state.stt_service.start_recording()
            st.session_state.is_recording = True
            # Only enable audio recording if the toggle is on
            st.session_state.is_recording_audio = st.session_state.get("audio_recording_enabled", True)
            st.session_state.recorded_audio_data = []  # Clear previous recording
            st.info("🎤 Recording started - speak now!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording):
            logger.info("Stop recording button clicked")
            st.session_state.stt_service.stop_recording()
            st.session_state.is_recording = False
            st.session_state.is_recording_audio = False  # Stop audio recording
            st.session_state.transcription_timeout = False
            
            # Save recorded audio to file
            if st.session_state.recorded_audio_data:
                filename = save_recorded_audio()
                if filename:
                    st.success(f"🎵 Audio saved as: {filename}")
                else:
                    st.warning("⚠️ Failed to save audio file")
            else:
                st.info("ℹ️ No audio data recorded")
            
            # Show audio buffer size for debugging
            stats = st.session_state.stt_service.get_recording_stats()
            st.info(f"📊 Audio buffer size: {stats.get('recorded_audio_size', 0)} bytes")

            # Commit audio for transcription
            with st.spinner("🔄 Processing audio..."):
                fut = threading.Event()
                result_holder = {}

                def _runner():
                    res = asyncio.run(st.session_state.stt_service.commit_audio_for_transcription())
                    result_holder["res"] = res
                    fut.set()

                t = threading.Thread(target=_runner, daemon=True)
                add_script_run_ctx(t)
                t.start()
                fut.wait()
                result = result_holder["res"]
    
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

    _pump_ui_events()

    # Drain raw mic chunks into session buffer when recording-to-file is enabled
    if st.session_state.get("is_recording_audio", False):
        drained = 0
        while True:
            try:
                chunk = raw_audio_q.get_nowait()
            except Empty:
                break
            st.session_state.recorded_audio_data.append(chunk)
            drained += 1
        if drained:
            logger.info(f"Buffered {drained} raw audio chunks for file recording")


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
    logger.info("🚀 Starting main application function")
    
    # Initialize session state
    logger.info("📝 Initializing session state")
    initialize_session_state()

    if "audio_consumer_started" not in st.session_state:
        t = threading.Thread(target=_audio_consumer_loop, daemon=True)
        add_script_run_ctx(t)  # safe to read session_state in the thread
        t.start()
        st.session_state.audio_consumer_started = True

    # Get configuration - handle missing secrets gracefully
    try:
        logger.info("⚙️ Loading configuration")
        config = get_config_from_secrets()
        logger.info(f"✅ Configuration loaded: {list(config.keys())}")
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        st.error(f"Configuration error: {e}")
        st.stop()

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
    # Only log STT interface rendering once per session
    if "stt_interface_rendered" not in st.session_state:
        logger.info("🎤 Rendering STT interface")
        st.session_state.stt_interface_rendered = True
    
    render_stt_interface()
    logger.info("✅ STT interface rendered")

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # make a fresh placeholder for this turn
        st.session_state.text_placeholder = st.empty()
        run_coro_in_thread(process_user_message(prompt, sidebar_config, config))
        # kick a first rerun so streaming can start updating
        st.session_state._needs_rerun = True

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

    # Trigger one rerun on the main thread if any callback asked for it
    if st.session_state.pop("_needs_rerun", False):
        # Important: stop WebRTC first to avoid timer races during rerun
        cleanup_webrtc_connections()
        st.rerun()



if __name__ == "__main__":
    main()

