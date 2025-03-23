"""
Basisklassen und Interfaces für Language Models im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Generator


class LLM(ABC):
    """
    Basisklasse für alle Language Models.

    Ermöglicht die Interaktion mit verschiedenen Language Models.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generiert eine Antwort basierend auf einem Prompt.

        Args:
            prompt: Der Prompt für das Language Model
            **kwargs: Weitere Parameter für das Language Model

        Returns:
            Die generierte Antwort als String
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        Generiert eine Antwort als Stream basierend auf einem Prompt.

        Args:
            prompt: Der Prompt für das Language Model
            **kwargs: Weitere Parameter für das Language Model

        Returns:
            Ein Generator, der die generierte Antwort stückweise zurückgibt
        """
        pass

    def format_prompt(self, query: str, context: str) -> str:
        """
        Formatiert einen Prompt für das RAG-System.

        Args:
            query: Die Benutzeranfrage
            context: Der Kontext aus dem Retrieval-Schritt

        Returns:
            Ein formatierter Prompt für das Language Model
        """
        return f"""Beantworte die folgende Frage basierend auf dem gegebenen Kontext.
Wenn die Antwort nicht im Kontext zu finden ist, antworte mit "Ich kann diese Frage nicht beantworten, da die nötigen Informationen nicht im Kontext enthalten sind."

Kontext:
{context}

Frage:
{query}

Antwort:"""