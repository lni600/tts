"""
Conversation zipper utility for bundling transcripts and audio files.
"""
import json
import zipfile
import io
from datetime import datetime
from typing import List, Dict, Any
import os


def build_conversation_zip(messages: List[Dict[str, Any]], output_dir: str = "temp") -> tuple[bytes, str]:
    """
    Create a ZIP file containing conversation transcript and audio files.
    
    Args:
        messages: List of conversation messages with audio paths
        output_dir: Directory to store temporary files
    
    Returns:
        Tuple of (zip_bytes, filename)
    """
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.zip"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add transcript.json
        transcript = {
            "conversation_id": f"conv_{timestamp}",
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
        
        for i, msg in enumerate(messages):
            message_data = {
                "role": msg["role"],
                "text": msg["text"],
                "started_at": msg["started_at"],
                "ended_at": msg["ended_at"]
            }
            
            # Add audio file reference if available
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                message_data["audio_file"] = f"audio/assistant_{i:04d}.wav"
                
                # Add audio file to ZIP
                audio_zip_path = f"audio/assistant_{i:04d}.wav"
                zip_file.write(msg["audio_path"], audio_zip_path)
            
            transcript["messages"].append(message_data)
        
        # Add transcript.json to ZIP
        zip_file.writestr("transcript.json", json.dumps(transcript, indent=2))
        
        # Add README
        readme_content = f"""Conversation Export
Generated: {datetime.now().isoformat()}
Total Messages: {len(messages)}

This ZIP contains:
- transcript.json: Complete conversation with timestamps
- audio/: WAV files for assistant responses

Audio Format: PCM16, 16kHz, Mono
"""
        zip_file.writestr("README.txt", readme_content)
    
    return zip_buffer.getvalue(), filename


def save_conversation_zip(messages: List[Dict[str, Any]], output_path: str) -> str:
    """
    Save conversation ZIP to disk.
    
    Args:
        messages: List of conversation messages
        output_path: Path to save the ZIP file
    
    Returns:
        Path to saved ZIP file
    """
    zip_bytes, filename = build_conversation_zip(messages)
    
    full_path = os.path.join(output_path, filename)
    
    with open(full_path, 'wb') as f:
        f.write(zip_bytes)
    
    return full_path


def cleanup_temp_files(messages: List[Dict[str, Any]]) -> None:
    """
    Clean up temporary audio files after creating ZIP.
    
    Args:
        messages: List of messages with audio paths
    """
    for msg in messages:
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            try:
                os.remove(msg["audio_path"])
            except OSError:
                pass  # Ignore errors during cleanup
