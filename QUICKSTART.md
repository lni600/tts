# 🚀 Quick Start Guide

## Get Running in 3 Steps

### 1. Install Dependencies
```bash
python3 setup.py
```

### 2. Configure API Keys
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your API keys
```

### 3. Run the App
```bash
streamlit run app.py
```

## 🎯 What You'll Get

- **Realtime TTS Tool** with ElevenLabs integration
- **WebRTC Audio Streaming** for low-latency audio
- **Streaming LLM Responses** with OpenAI
- **Conversation Export** as ZIP files with transcripts and audio
- **Production-Ready Architecture** with error handling

## 🔧 Configuration Options

### Required Configuration
```toml
ELEVENLABS_API_KEY = "sk-your-key"
ELEVENLABS_VOICE_ID = "your-voice-id"
OPENAI_API_KEY = "sk-your-openai-key"
```

## 🎵 Audio Features

- **Sample Rate**: 16,000 Hz (mono)
- **Format**: PCM16
- **Chunk Size**: 20ms frames
- **Latency**: ~300-600ms from first token to audio

## 📱 Usage

1. Click "🔄 Initialize Services"
2. Type your message in the chat input
3. Watch text stream in real-time
4. Listen to audio as it's generated
5. Click "💾 Save Conversation" to download ZIP

## 🐛 Troubleshooting

- **No Audio**: Check browser permissions and WebRTC support
- **Connection Errors**: Verify API keys and network access
- **Performance Issues**: Adjust chunk sizes in secrets.toml

## 📚 Full Documentation

See [README.md](README.md) for complete details and advanced configuration.

---

**Ready to chat? Run `streamlit run app.py` and start talking! 🎤✨**
