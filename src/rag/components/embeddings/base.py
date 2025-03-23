"""
Basisklassen und Interfaces für Embedding-Modelle im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from ...components.data_sources.base import Document


class Embedder(ABC):
    """
    Basisklasse für alle Embedding-Modelle.

    Ermöglicht das Erstellen von Vektorrepräsentationen von Texten.
    """


    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Erstellt ein Embedding für eine einzelne Query.

        Args:
            text: Der zu embedende Text

        Returns:
            Ein Embedding-Vektor als Liste von Floats
        """
        pass


    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Erstellt Embeddings für eine Liste von Dokumenten.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Embedding-Vektoren
        """
        pass


    def _get_document_texts(self, documents: List[Document]) -> List[str]:
        """
        Extrahiert den Textinhalt aus einer Liste von Dokumenten.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Texten
        """
        return [doc.content for doc in documents]