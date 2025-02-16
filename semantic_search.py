import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class SemanticSearchEngine:
    """
    A semantic search engine that uses BERT embeddings to find semantically similar documents.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the semantic search engine.

        Args:
            model_name (str): Name of the pre-trained SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = None
        self.documents = None

    def index(self, documents):
        """
        Index a list of documents by computing their embeddings.

        Args:
            documents (list[str]): List of documents to index.
        """
        if not documents or not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
            raise ValueError("Documents must be a non-empty list of strings.")
        
        logging.info("Indexing documents...")
        self.documents = documents
        self.document_embeddings = self.model.encode(documents, convert_to_tensor=True)
        logging.info(f"Indexed {len(documents)} documents.")

    def search(self, query, top_k=5):
        """
        Perform a semantic search for the given query.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: A list of dictionaries containing the document and its similarity score.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
        
        if self.document_embeddings is None or self.documents is None:
            raise ValueError("No documents have been indexed. Call the `index` method first.")
        
        logging.info(f"Performing search for query: {query}")
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "document": self.documents[idx],
                "score": score.item()
            })
        
        logging.info(f"Found {len(results)} results.")
        
        return results
