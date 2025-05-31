from math import ceil
import torch
import gc
import numpy as np
from typing import List
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import LocalEmbeddingFunction

class VectorStoreManager:
    def __init__(self, embedding_function):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_function: Function to generate embeddings
        """
        self.embedding_function = embedding_function

    def create_vecdb(self, path: str, dataset: List, batch_size: int = 64, create_new: bool = True) -> Chroma:
        """
        Create or update a Chroma vector database with batch processing.
        
        Args:
            path (str): Directory path to persist the vector database
            dataset (List): List of documents or text data
            batch_size (int): Number of items to process per batch
            create_new (bool): Whether to create a new vector database or update an existing one
            
        Returns:
            Chroma: Vector store instance
        """
        if create_new:
            vectorstore = None
            print("Creating a new Vector Database...")
            batches = self._batchify(dataset, batch_size)
            total_batches = ceil(len(dataset) / batch_size)

            for batch_idx, batch in enumerate(batches):
                print(f"Processing batch {batch_idx + 1}/{total_batches}...")
                self._clear_memory()

                if vectorstore:
                    batch_embeddings = [self.embedding_function(text) for text in batch]
                    vectorstore.add_texts(texts=batch, embeddings=batch_embeddings)
                else:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embedding_function,
                        persist_directory=path
                    )

                self._clear_memory()
                vectorstore.persist()
                print(f"Batch {batch_idx + 1}/{total_batches} saved.")

            print("Vector Database Created and Saved Successfully!")
            return vectorstore
        else:
            vectorstore = Chroma(
                embedding_function=self.embedding_function,
                persist_directory=path
            )
            print("Loaded existing Vector Database...")
            return vectorstore

    def query_vecdb(self, query: str, vectorstore: Chroma) -> List:
        """
        Query the vector database.
        
        Args:
            query (str): Query text
            vectorstore (Chroma): Vector store instance
            
        Returns:
            List: Retrieved documents
        """
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.60, "k": 3}
        )
        return retriever.invoke(query)

    def _batchify(self, data: List, batch_size: int):
        """Helper method to create batches."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def _clear_memory(self):
        """Helper method to clear memory."""
        gc.collect()
        torch.cuda.empty_cache()

class SimilarityCalculator:
    def __init__(self, model):
        """
        Initialize the similarity calculator with a model.
        
        Args:
            model: The model to use for generating embeddings
        """
        self.embedding_function = LocalEmbeddingFunction(model)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score
        """
        embeddings1 = np.array(self.embedding_function.embed_query(query=text1))
        embeddings2 = np.array(self.embedding_function.embed_query(query=text2))

        embeddings1 = embeddings1.reshape(1, -1)
        embeddings2 = embeddings2.reshape(1, -1)

        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return similarity