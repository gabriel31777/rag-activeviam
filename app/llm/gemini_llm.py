"""Gemini LLM implementation using Google GenAI."""

from __future__ import annotations

from typing import Optional

from google import genai
from google.genai import types

from llm.base_llm import BaseLLM
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()


class GeminiLLM(BaseLLM):
    """LLM provider using Google Gemini API."""

    def __init__(self, model_name: str = "gemma-3-27b-it"):
        if not config.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file."
            )

        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model_name = model_name
        logger.info("GeminiLLM initialized: %s", model_name)

    def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using Gemini.

        Args:
            prompt: The user/task prompt.
            system_prompt: Optional system instruction for grounding.

        Returns:
            Generated text.
        """
        try:
            gen_config = types.GenerateContentConfig(
                max_output_tokens=config.max_tokens,
                temperature=0.2,  # Low for factual accuracy
            )
            
            # Gemma models don't support system_instruction parameter
            # So we prepend it to the user prompt instead
            final_prompt = prompt
            if system_prompt:
                if self.model_name.startswith("gemma"):
                    # For Gemma: prepend system prompt to user prompt
                    final_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    # For Gemini: use native system_instruction
                    gen_config.system_instruction = system_prompt

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt,
                config=gen_config,
            )
            return response.text
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return f"Error generating response: {e}"
