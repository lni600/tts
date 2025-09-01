# 🎤 Streamlit Realtime TTS Chatbot

A production-ready Streamlit chatbot with realtime Text-to-Speech using ElevenLabs, featuring WebRTC audio streaming and conversation management.

## ✨ Features

- **Realtime TTS**: Stream audio as text is generated using ElevenLabs WebSocket API
- **WebRTC Audio**: Low-latency audio streaming via browser WebRTC
- **Streaming LLM**: Token-by-token text generation with OpenAI or dummy LLM
- **Conversation Management**: Save conversations as ZIP files with transcripts and audio
- **Modular Architecture**: Easy to switch between different TTS and LLM providers
- **Production Ready**: Error handling, reconnection logic, and graceful cleanup

## 🏗️ Architecture

```
app.py (Main Streamlit App)
├── llm/streaming_llm.py (LLM Integration)
├── realtime/tts_elevenlabs_ws.py (TTS WebSocket Client)
├── realtime/audio_sender.py (WebRTC Audio Track)
├── utils/audio.py (Audio Processing)
└── utils/zipper.py (Conversation Export)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

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

# OpenAI Configuration (optional)
OPENAI_API_KEY = "sk-your-openai-key"

# Feature Flags
USE_DUMMY_LLM = true
USE_DUMMY_TTS = true
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
export USE_DUMMY_LLM="true"
export USE_DUMMY_TTS="true"
```

### TTS Settings

- **Sample Rate**: 16,000 Hz (mono)
- **Chunk Size**: 20ms frames
- **Audio Format**: PCM16
- **Buffer Strategy**: Time-based (150ms) + punctuation-triggered flush

## 🎯 Usage

### 1. Initialize Services

Click the "🔄 Initialize Services" button to connect to TTS and LLM services.

### 2. Start Chatting

Type your message in the chat input. The assistant will:
- Generate streaming text response
- Convert text to speech in real-time
- Stream audio via WebRTC
- Save audio files for each response

### 3. Save Conversations

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

#### Dummy TTS (Testing)
- Generates sine wave audio
- No API key required
- Simulates real-time behavior

### LLM Providers

#### OpenAI (Production)
- GPT-4o-mini streaming
- Real conversation capabilities
- Requires API key

#### Dummy LLM (Testing)
- Generates canned responses
- Simulates streaming behavior
- No API key required

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
USE_DUMMY_LLM=false
USE_DUMMY_TTS=false
ELEVENLABS_API_KEY=your-production-key
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
- [aiortc](https://github.com/aiortc/aiortc) for WebRTC implementation
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for Streamlit WebRTC integration

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the debug info in the app
3. Open an issue on GitHub
4. Check ElevenLabs documentation for API-specific issues

---

**Happy Chatting! 🎤✨**
