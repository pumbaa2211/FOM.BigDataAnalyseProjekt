"""
Basisklassen und Interfaces für Vector Stores im RAG-System.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

from ...components.data_sources.base import Document


class VectorStore(ABC):
    """
    Basisklasse für alle Vector Stores.

    Ermöglicht das Speichern und Abrufen von Dokumenten basierend auf Vektorähnlichkeit.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Fügt Dokumente und ihre Embeddings zum Vector Store hinzu.

        Args:
            documents: Eine Liste von Document-Objekten
            embeddings: Eine Liste von Embedding-Vektoren
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Führt eine Ähnlichkeitssuche im Vector Store durch.

        Args:
            query_embedding: Das Embedding der Query
            k: Die Anzahl der zurückzugebenden ähnlichsten Dokumente
            threshold: Optional ein Schwellenwert für die Ähnlichkeit

        Returns:
            Eine Liste von Tupeln (Document, Ähnlichkeitswert)
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Löscht alle Dokumente und Embeddings aus dem Vector Store."""
        pass

    @property
    @abstractmethod
    def document_count(self) -> int:
        """
        Gibt die Anzahl der im Vector Store gespeicherten Dokumente zurück.

        Returns:
            Anzahl der Dokumente
        """
        pass