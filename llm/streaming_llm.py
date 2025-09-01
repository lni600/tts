"""
Streaming LLM integration for real-time text generation.
"""
import asyncio
import os
from typing import AsyncGenerator, Optional
import openai
from datetime import datetime


class StreamingLLM:
    """Base class for streaming language models."""
    
    async def stream_assistant_reply(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream assistant reply tokens.
        
        Args:
            prompt: User input prompt
            
        Yields:
            Text tokens as they become available
        """
        raise NotImplementedError


class DummyStreamingLLM(StreamingLLM):
    """Dummy LLM for testing without API credentials."""
    
    def __init__(self, delay_ms: float = 50):
        self.delay_ms = delay_ms / 1000.0  # Convert to seconds
        
    async def stream_assistant_reply(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream a dummy response with simulated latency.
        
        Args:
            prompt: User input prompt
            
        Yields:
            Text tokens with delays
        """
        # Create a dummy response based on the prompt
        responses = [
            "I understand your question about",
            f"'{prompt}'. ",
            "Let me think about this... ",
            "Based on my analysis, ",
            "I believe the answer is that ",
            "this is a fascinating topic. ",
            "There are several aspects to consider: ",
            "First, we should look at the context. ",
            "Second, there are practical implications. ",
            "Finally, this leads us to conclude that ",
            "your question touches on important concepts. ",
            "I hope this helps clarify things!"
        ]
        
        for token in responses:
            yield token
            await asyncio.sleep(self.delay_ms)


class OpenAIStreamingLLM(StreamingLLM):
    """OpenAI streaming LLM integration."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def stream_assistant_reply(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response from OpenAI API.
        
        Args:
            prompt: User input prompt
            
        Yields:
            Text tokens from OpenAI
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.7,
                max_tokens=500
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"


def create_streaming_llm(use_dummy: bool = True, openai_api_key: Optional[str] = None) -> StreamingLLM:
    """
    Factory function to create appropriate streaming LLM.
    
    Args:
        use_dummy: Whether to use dummy LLM
        openai_api_key: OpenAI API key if using real LLM
        
    Returns:
        StreamingLLM instance
    """
    if use_dummy:
        return DummyStreamingLLM()
    elif openai_api_key:
        return OpenAIStreamingLLM(openai_api_key)
    else:
        # Fallback to dummy if no API key provided
        return DummyStreamingLLM()


async def test_streaming():
    """Test function for streaming LLM."""
    llm = DummyStreamingLLM()
    
    async for token in llm.stream_assistant_reply("Hello, how are you?"):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(test_streaming())
