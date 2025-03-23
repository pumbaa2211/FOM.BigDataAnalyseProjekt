"""
Implementiert OpenAI-basierte Embeddings für das RAG-System.
"""

import os
from typing import List
from openai import OpenAI

from ...config import config
from ...components.data_sources.base import Document
from .base import Embedder


class OpenAIEmbedder(Embedder):
    """
    Implementiert Embeddings mit OpenAI.
    """

    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialisiert den OpenAIEmbedder.

        Args:
            model: Das zu verwendende Embedding-Modell
            api_key: Der OpenAI API-Key
        """
        self.model = model or config.embedding.model
        self.api_key = api_key or config.embedding.api_key
        self.client = OpenAI(api_key=self.api_key)

    def embed_query(self, text: str) -> List[float]:
        """
        Erstellt ein Embedding für eine einzelne Query mit OpenAI.

        Args:
            text: Der zu embedende Text

        Returns:
            Ein Embedding-Vektor als Liste von Floats
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Erstellt Embeddings für eine Liste von Dokumenten mit OpenAI.

        Args:
            documents: Eine Liste von Document-Objekten

        Returns:
            Eine Liste von Embedding-Vektoren
        """
        texts = self._get_document_texts(documents)

        # OpenAI hat ein Limit für die Anzahl der Tokens, die in einem API-Call
        # verarbeitet werden können. Daher teilen wir die Dokumente in Batches auf.
        batch_size = 50
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                # Bei einem Fehler versuchen wir, die Dokumente einzeln zu embedden
                print(f"Fehler beim Embedden von Batch {i}-{i+batch_size}: {e}")
                print("Versuche, Dokumente einzeln zu embedden...")

                for j in range(i, min(i+batch_size, len(texts))):
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=texts[j]
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as e:
                        print(f"Fehler beim Embedden von Dokument {j}: {e}")
                        # Dummy-Embedding erstellen, um die Reihenfolge beizubehalten
                        # Dies ist nicht ideal, aber besser als das gesamte Embedding abzubrechen
                        embeddings.append([0.0] * 1536)  # OpenAI Embeddings haben 1536 Dimensionen

        return embeddings