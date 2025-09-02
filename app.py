"""
Streamlit Chatbot with Realtime TTS using ElevenLabs.
"""
import streamlit as st
import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

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
        "use_dummy_llm": str(st.secrets.get("USE_DUMMY_LLM", os.getenv("USE_DUMMY_LLM", "true"))).lower() == "true",
        "use_dummy_tts": str(st.secrets.get("USE_DUMMY_TTS", os.getenv("USE_DUMMY_TTS", "true"))).lower() == "true",
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
        index=0 if config["elevenlabs_voice_id"] in config["available_voices"] else 0,
        key="voice_selectbox"
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




async def process_user_message(user_input: str, sidebar_config: Dict[str, Any], config: Dict[str, Any]):
    """Process user message and generate streaming response with synchronized text + audio."""
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
        # Create fresh clients inside this loop
        tts_client = create_tts_client(
            voice_id=sidebar_config["selected_voice"],
            api_key=config["elevenlabs_api_key"],
            use_dummy=sidebar_config["use_dummy_tts"],
            sample_rate=config["tts_sample_rate"],
        )
        llm = create_streaming_llm(
            use_dummy=sidebar_config["use_dummy_llm"],
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

        # Audio collection task
        async def collect_audio():
            try:
                chunk_count = 0
                total_bytes = 0
                async for chunk in tts_client.audio_chunks():
                    st.session_state.current_audio_chunks.append(chunk)
                    st.session_state.audio_queue_manager.put_audio(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    # Audio chunk collected
                
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
            audio_path = save_audio_chunks(st.session_state.current_audio_chunks)
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
        if audio_path:
            st.audio(audio_path, format="audio/wav")
        else:
            st.warning("⚠️ Audio was generated but could not be saved/displayed")

    except Exception as e:
        st.error(f"Error processing message: {e}")
        # Clean up TTS client on error
        try:
            if 'tts_client' in locals() and hasattr(tts_client, "close"):
                await tts_client.close()
        except:
            pass


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
    
    audio_queue = st.session_state.audio_queue_manager.create_queue()
    track = get_track(audio_queue, sample_rate=16000, chunk_duration_ms=20)

    # Create WebRTC streamer
    webrtc_streamer(
        key="audio-player",
        mode=WebRtcMode.RECVONLY,
        frontend_rtc_configuration=rtc_configuration,
        server_rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": False, "audio": True},
        source_audio_track=track,
        async_processing=True,
    )


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

    # Render chat history
    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if True:  # We always recreate services per message
            try:
                asyncio.run(process_user_message(prompt, sidebar_config, config))
            except Exception as e:
                st.error(f"Error processing message: {e}")
        else:
            st.error("Please initialize services first!")

    # WebRTC audio player
    render_webrtc_player()

    # System status
    with st.expander("🔧 System Status"):
        st.json({
            "connection_status": st.session_state.connection_status,
            "message_count": len(st.session_state.messages),
            "audio_chunks_ready": len(st.session_state.current_audio_chunks) > 0,
        })



if __name__ == "__main__":
    main()
