"""
Implementiert einen einfachen Retriever für das RAG-System.
"""

from typing import List, Optional

from ...config import config
from ..data_sources.base import Document
from ..embeddings.base import Embedder
from ..vector_stores.base import VectorStore
from .base import Retriever


class SimpleRetriever(Retriever):
    """
    Implementiert einen einfachen Retriever für das RAG-System.

    Dieser Retriever nutzt einen Vector Store, um relevante Dokumente abzurufen.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = None,
        threshold: Optional[float] = None
    ):
        """
        Initialisiert den SimpleRetriever.

        Args:
            vector_store: Der zu verwendende Vector Store
            embedder: Der zu verwendende Embedder
            top_k: Die Anzahl der zurückzugebenden relevantesten Dokumente
            threshold: Optional ein Schwellenwert für die Relevanz
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k or config.retrieval.top_k
        self.threshold = threshold if threshold is not None else config.retrieval.threshold

    def retrieve(self, query: str, top_k: int = None, threshold: Optional[float] = None) -> List[Document]:
        """
        Ruft relevante Dokumente basierend auf einer Query ab.

        Args:
            query: Die Benutzeranfrage
            top_k: Die Anzahl der zurückzugebenden relevantesten Dokumente
            threshold: Optional ein Schwellenwert für die Relevanz

        Returns:
            Eine Liste von relevanten Dokumenten
        """
        # Fallback auf Standardwerte, wenn keine Parameter angegeben wurden
        top_k = top_k or self.top_k
        threshold = threshold if threshold is not None else self.threshold

        # Query embedden
        query_embedding = self.embedder.embed_query(query)

        # Relevante Dokumente abrufen
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            threshold=threshold
        )

        # Nur die Dokumente zurückgeben, ohne die Ähnlichkeitswerte
        documents = [doc for doc, _ in results]

        return documents

    def format_retrieved_documents(self, documents: List[Document]) -> str:
        """
        Formatiert die abgerufenen Dokumente für die Verwendung im RAG-System.

        Args:
            documents: Die abgerufenen Dokumente

        Returns:
            Ein formatierter String, der die Dokumente enthält
        """
        if not documents:
            return "Es wurden keine relevanten Dokumente gefunden."

        formatted_texts = []

        for i, doc in enumerate(documents):
            source_info = f"Quelle: {doc.metadata.get('source', 'Unbekannt')}" if doc.metadata else "Quelle: Unbekannt"
            formatted_texts.append(f"[Dokument {i+1}] {source_info}\n{doc.content}\n")

        return "\n".join(formatted_texts)