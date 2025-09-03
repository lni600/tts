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


def create_streaming_llm(openai_api_key: str) -> StreamingLLM:
    """
    Factory function to create streaming LLM.
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        StreamingLLM instance
    """
    if not openai_api_key:
        raise ValueError("OpenAI API key is required")
    return OpenAIStreamingLLM(openai_api_key)


async def test_streaming():
    """Test function for streaming LLM."""
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    llm = create_streaming_llm(api_key)
    
    async for token in llm.stream_assistant_reply("Hello, how are you?"):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(test_streaming())
