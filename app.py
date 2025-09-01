"""
Streamlit Chatbot with Realtime TTS using ElevenLabs.
"""
import streamlit as st
import asyncio
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# Import our modules
from llm.streaming_llm import create_streaming_llm
from realtime.tts_elevenlabs_ws import create_tts_client
from realtime.audio_sender import get_track, AudioQueueManager
from utils.audio import write_wav
from utils.zipper import build_conversation_zip

# WebRTC imports
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av


# Page configuration
st.set_page_config(
    page_title="Realtime TTS Chatbot",
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
    
    if "tts_client" not in st.session_state:
        st.session_state.tts_client = None
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "audio_queue_manager" not in st.session_state:
        st.session_state.audio_queue_manager = AudioQueueManager()
    
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = "disconnected"
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None


def get_config_from_secrets():
    """Get configuration from Streamlit secrets or environment variables."""
    config = {
        "elevenlabs_api_key": st.secrets.get("ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY", "")),
        "elevenlabs_voice_id": st.secrets.get("ELEVENLABS_VOICE_ID", os.getenv("ELEVENLABS_VOICE_ID", "default_voice")),
        "openai_api_key": st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        "use_dummy_llm": st.secrets.get("USE_DUMMY_LLM", os.getenv("USE_DUMMY_LLM", "true")).lower() == "true",
        "use_dummy_tts": st.secrets.get("USE_DUMMY_TTS", os.getenv("USE_DUMMY_TTS", "true")).lower() == "true",
        "available_voices": st.secrets.get("AVAILABLE_VOICES", "default_voice").split(","),
        "tts_sample_rate": int(st.secrets.get("TTS_SAMPLE_RATE", "16000")),
        "tts_chunk_size_ms": int(st.secrets.get("TTS_CHUNK_SIZE_MS", "20"))
    }
    return config


def render_header():
    """Render the main header with status."""
    st.markdown('<h1 class="main-header">🎤 Realtime TTS Chatbot</h1>', unsafe_allow_html=True)
    
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
        index=0 if config["elevenlabs_voice_id"] in config["available_voices"] else 0
    )
    
    # Feature toggles
    use_dummy_llm = st.sidebar.checkbox("Use Dummy LLM", value=config["use_dummy_llm"])
    use_dummy_tts = st.sidebar.checkbox("Use Dummy TTS", value=config["use_dummy_tts"])
    
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
        "selected_voice": selected_voice,
        "use_dummy_llm": use_dummy_llm,
        "use_dummy_tts": use_dummy_tts
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


async def initialize_services(config: Dict[str, Any], sidebar_config: Dict[str, Any]):
    """Initialize TTS and LLM services."""
    try:
        # Initialize TTS client
        if not st.session_state.tts_client or st.session_state.connection_status != "connected":
            st.session_state.connection_status = "connecting"
            
            tts_client = create_tts_client(
                voice_id=sidebar_config["selected_voice"],
                api_key=config["elevenlabs_api_key"],
                use_dummy=sidebar_config["use_dummy_tts"],
                sample_rate=config["tts_sample_rate"]
            )
            
            if await tts_client.connect():
                st.session_state.tts_client = tts_client
                st.session_state.connection_status = "connected"
            else:
                st.session_state.connection_status = "error"
                return False
        
        # Initialize LLM
        if not st.session_state.llm:
            st.session_state.llm = create_streaming_llm(
                use_dummy=sidebar_config["use_dummy_llm"],
                openai_api_key=config["openai_api_key"]
            )
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        st.session_state.connection_status = "error"
        return False


async def process_user_message(user_input: str, sidebar_config: Dict[str, Any]):
    """Process user message and generate streaming response."""
    if not st.session_state.tts_client or not st.session_state.llm:
        st.error("Services not initialized!")
        return
    
    # Add user message to history
    user_message = {
        "role": "user",
        "text": user_input,
        "started_at": datetime.now().isoformat(),
        "ended_at": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_message)
    
    # Clear current audio chunks
    st.session_state.current_audio_chunks = []
    
    # Create placeholder for streaming text
    text_placeholder = st.empty()
    current_text = ""
    
    # Start TTS audio collection
    audio_collection_task = asyncio.create_task(collect_audio_chunks())
    
    try:
        # Stream LLM response
        async for token in st.session_state.llm.stream_assistant_reply(user_input):
            current_text += token
            text_placeholder.markdown(f"**Assistant:** {current_text}")
            
            # Send token to TTS
            await st.session_state.tts_client.send_text_fragment(token)
        
        # Finalize TTS
        await st.session_state.tts_client.finalize()
        
        # Wait for audio collection to complete
        await audio_collection_task
        
        # Save audio file
        if st.session_state.current_audio_chunks:
            audio_path = save_audio_chunks(st.session_state.current_audio_chunks)
            
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "text": current_text,
                "audio_path": audio_path,
                "started_at": datetime.now().isoformat(),
                "ended_at": datetime.now().isoformat()
            }
            st.session_state.messages.append(assistant_message)
            
            # Update the placeholder with final message
            text_placeholder.markdown(f"**Assistant:** {current_text}")
            
            # Show audio player
            st.audio(audio_path, format="audio/wav")
        
    except Exception as e:
        st.error(f"Error processing message: {e}")
        if not audio_collection_task.done():
            audio_collection_task.cancel()


async def collect_audio_chunks():
    """Collect audio chunks from TTS client."""
    try:
        async for chunk in st.session_state.tts_client.audio_chunks():
            st.session_state.current_audio_chunks.append(chunk)
            
            # Put chunk in audio queue for WebRTC
            if st.session_state.audio_queue_manager:
                st.session_state.audio_queue_manager.put_audio(chunk)
                
    except Exception as e:
        st.error(f"Error collecting audio: {e}")


def save_audio_chunks(chunks: List[bytes]) -> str:
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
        write_wav(chunks, filepath, sample_rate=16000)
        
        return filepath
        
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


def render_webrtc_player():
    """Render WebRTC audio player."""
    st.markdown("### 🔊 Audio Player")
    
    # WebRTC configuration
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    })
    
    # Create WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="audio-player",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )
    
    # Attach audio track when playing
    if webrtc_ctx.state.playing:
        if not hasattr(webrtc_ctx, 'audio_track_attached'):
            # Create audio track from queue
            audio_queue = st.session_state.audio_queue_manager.create_queue()
            audio_track = get_track(audio_queue, sample_rate=16000, chunk_duration_ms=20)
            
            # Add track to peer connection
            webrtc_ctx.addTrack(audio_track)
            webrtc_ctx.audio_track_attached = True


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Get configuration
    config = get_config_from_secrets()
    
    # Render header
    render_header()
    
    # Render sidebar
    sidebar_config = render_sidebar(config)
    
    # Initialize services
    if st.button("🔄 Initialize Services"):
        with st.spinner("Initializing services..."):
            success = asyncio.run(initialize_services(config, sidebar_config))
            if success:
                st.success("Services initialized successfully!")
            else:
                st.error("Failed to initialize services!")
    
    # Render chat history
    render_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if st.session_state.connection_status == "connected":
            # Process message asynchronously
            asyncio.run(process_user_message(prompt, sidebar_config))
        else:
            st.error("Please initialize services first!")
    
    # WebRTC audio player
    render_webrtc_player()
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.json({
            "connection_status": st.session_state.connection_status,
            "message_count": len(st.session_state.messages),
            "current_audio_chunks": len(st.session_state.current_audio_chunks),
            "tts_client_connected": st.session_state.tts_client is not None,
            "llm_initialized": st.session_state.llm is not None
        })


if __name__ == "__main__":
    main()
