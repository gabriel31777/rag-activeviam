"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Generic interface for language model providers."""

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt.
            system_prompt: Optional system-level instruction.

        Returns:
            Generated text.
        """
        ...
