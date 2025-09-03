# 🎤 Streamlit Realtime TTS & STT Chatbot

A production-ready Streamlit chatbot with realtime Text-to-Speech and Speech-to-Text capabilities, featuring WebRTC audio streaming, OpenAI Realtime API transcription, and conversation management.

## ✨ Features

- **Realtime TTS**: Stream audio as text is generated using ElevenLabs WebSocket API
- **Speech-to-Text**: Voice input with OpenAI Realtime API for accurate transcription
- **WebRTC Audio**: Low-latency audio streaming and microphone capture via browser WebRTC
- **Streaming LLM**: Token-by-token text generation with OpenAI
- **Conversation Management**: Save conversations as ZIP files with transcripts and audio
- **Modular Architecture**: Easy to switch between different TTS, STT, and LLM providers
- **Production Ready**: Error handling, reconnection logic, and graceful cleanup

## 🏗️ Architecture

```
app.py (Main Streamlit App)
├── llm/streaming_llm.py (LLM Integration)
├── realtime/tts_elevenlabs_ws.py (TTS WebSocket Client)
├── realtime/audio_sender.py (WebRTC Audio Track)
├── stt/openai_realtime_stt.py (Speech-to-Text with OpenAI Realtime API)
├── stt/audio_receiver.py (WebRTC Microphone Capture)
├── stt/stt_service.py (STT Service Integration)
├── utils/audio.py (Audio Processing)
└── utils/zipper.py (Conversation Export)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The app uses OpenAI's Realtime API for speech-to-text, which requires an OpenAI API key with access to the Realtime API.

### 2. Configure Secrets

Copy the example secrets file and configure your API keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
# ElevenLabs Configuration
ELEVENLABS_API_KEY = "sk-your-actual-api-key"
ELEVENLABS_VOICE_ID = "your-voice-id"

# OpenAI Configuration (required for STT)
OPENAI_API_KEY = "sk-your-openai-key"

# Required API Keys
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

You can also set configuration via environment variables:

```bash
export ELEVENLABS_API_KEY="sk-your-key"
export ELEVENLABS_VOICE_ID="your-voice-id"

```

### Audio Settings

#### TTS Settings
- **Sample Rate**: 16,000 Hz (mono)
- **Chunk Size**: 20ms frames
- **Audio Format**: PCM16
- **Buffer Strategy**: Time-based (150ms) + punctuation-triggered flush

#### STT Settings
- **Sample Rate**: 24,000 Hz (mono) - OpenAI Realtime API requirement
- **Audio Format**: 16-bit PCM
- **Model**: gpt-4o-realtime-preview-2024-10-01 (configurable)
- **Transcription**: Real-time streaming via WebSocket
- **Voice Activity Detection**: Server-side VAD with configurable thresholds

## 🎯 Usage

### 1. Initialize Services

Click the "🔄 Initialize Services" button to connect to TTS and LLM services.

### 2. Start Chatting

You can interact with the chatbot in two ways:

#### Text Input
Type your message in the chat input. The assistant will:
- Generate streaming text response
- Convert text to speech in real-time
- Stream audio via WebRTC
- Save audio files for each response

#### Voice Input
Use the speech-to-text feature:
- Click "🎤 Start Recording" to begin voice capture
- Speak your message into the microphone
- Click "⏹️ Stop Recording" to end recording
- Review the transcribed text
- Click "📤 Send as Message" to send it to the chatbot

### 3. Voice Conversation Workflow

The app now supports complete voice conversations:

1. **Speak to the Bot**: Use the microphone to record your message
2. **Automatic Transcription**: Whisper converts your speech to text
3. **LLM Processing**: The chatbot processes your transcribed message
4. **Voice Response**: The bot responds with both text and synthesized speech
5. **Continuous Conversation**: Repeat the cycle for natural voice interaction

### 4. Save Conversations

Click "💾 Save Conversation" in the sidebar to download a ZIP file containing:
- `transcript.json` with timestamps and message history
- `audio/` folder with WAV files for each assistant response
- `README.txt` with export information

## 🔌 Integration Options

### TTS Providers

#### ElevenLabs (Production)
- WebSocket streaming API
- Low-latency audio generation
- Multiple voice options
- Requires API key



### STT Providers

#### OpenAI Realtime API (Production)
- Real-time speech recognition with low latency
- High-accuracy transcription using Whisper-1 model
- Supports multiple languages
- Requires OpenAI API key with Realtime API access
- WebSocket-based streaming for real-time processing

### LLM Providers

#### OpenAI (Production)
- GPT-4o-mini streaming
- Real conversation capabilities
- Requires API key



## 🌐 WebRTC Configuration

### STUN Servers

The app uses Google's public STUN servers:
- `stun:stun.l.google.com:19302`
- `stun:stun1.l.google.com:19302`

### Cloud Deployment

For cloud deployment, ensure your proxy allows:
- WebSocket connections
- UDP traffic for ICE/STUN
- HTTPS for secure connections

## 🐛 Troubleshooting

### No Audio Output

1. **Check Browser Permissions**: Ensure microphone access is allowed
2. **WebRTC Issues**: Try refreshing the page
3. **STUN/TURN**: Check if your network blocks UDP traffic
4. **Browser Compatibility**: Test with Chrome/Edge (best WebRTC support)

### STT Issues

1. **Microphone Access**: Ensure browser has microphone permissions
2. **OpenAI API Key**: Verify you have access to the Realtime API
3. **Audio Quality**: Speak clearly and reduce background noise
4. **Connection**: Check WebSocket connection to OpenAI Realtime API
5. **Sample Rate**: Ensure audio is properly resampled to 24kHz

### Connection Errors

1. **API Keys**: Verify ElevenLabs API key is valid
2. **Voice ID**: Check if voice ID exists in your account
3. **Network**: Ensure outbound WebSocket connections are allowed
4. **Rate Limits**: Check ElevenLabs API usage limits

### Performance Issues

1. **Chunk Size**: Adjust `TTS_CHUNK_SIZE_MS` in secrets
2. **Buffer Size**: Modify queue sizes in `audio_sender.py`
3. **Sample Rate**: Consider lower sample rates for slower connections

## 📁 File Structure

```
tts/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .streamlit/
│   └── secrets.toml.example      # Configuration template
├── llm/
│   ├── __init__.py               # LLM package
│   └── streaming_llm.py          # Streaming LLM implementations
├── realtime/
│   ├── __init__.py               # Realtime package
│   ├── tts_elevenlabs_ws.py     # ElevenLabs TTS client
│   └── audio_sender.py           # WebRTC audio streaming
├── stt/
│   ├── __init__.py               # STT package
│   ├── openai_realtime_stt.py    # OpenAI Realtime STT client
│   ├── audio_receiver.py         # WebRTC microphone capture
│   └── stt_service.py            # STT service integration
└── utils/
    ├── __init__.py               # Utils package
    ├── audio.py                  # Audio processing utilities
    └── zipper.py                 # Conversation export utilities
```

## 🔒 Security Considerations

- **API Keys**: Never commit API keys to version control
- **WebRTC**: Uses public STUN servers (consider private servers for production)
- **Audio Data**: Audio chunks are temporarily stored in memory/disk
- **Network**: Ensure HTTPS in production environments

## 🚀 Production Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set secrets in Streamlit Cloud dashboard
4. Deploy

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Environment Variables

Set production environment variables:
```bash
ELEVENLABS_API_KEY=your-production-key
ELEVENLABS_VOICE_ID=your-voice-id
OPENAI_API_KEY=your-openai-key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [ElevenLabs](https://elevenlabs.io/) for TTS API
- [OpenAI Realtime API](https://openai.com/index/introducing-the-realtime-api/) for speech recognition
- [aiortc](https://github.com/aiortc/aiortc) for WebRTC implementation
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for Streamlit WebRTC integration

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the debug info in the app
3. Open an issue on GitHub
4. Check ElevenLabs documentation for API-specific issues

---

**Happy Voice Chatting! 🎤🎧✨**
