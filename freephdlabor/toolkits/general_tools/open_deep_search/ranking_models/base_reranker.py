from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Union

class BaseSemanticSearcher(ABC):
    """
    Abstract base class for semantic search implementations.
    
    This class defines the interface that all semantic searchers must implement.
    Subclasses should implement the _get_embeddings method according to their
    specific embedding source.
    """
    
    @abstractmethod
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings shape: (num_texts, embedding_dim)
        """
        pass

    def calculate_scores(
        self,
        queries: List[str],
        documents: List[str],
        normalize: str = "softmax"  # Options: "softmax", "scale", "none"
    ) -> torch.Tensor:
        """
        Calculate similarity scores between queries and documents.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            normalize: Normalization method:
                      - "softmax": Apply softmax normalization (default)
                      - "scale": Scale to 0-100 range
                      - "none": No normalization
            
        Returns:
            torch.Tensor of shape (num_queries, num_documents) containing similarity scores
        """
        # Get embeddings for queries and documents
        query_embeddings = self._get_embeddings(queries)
        doc_embeddings = self._get_embeddings(documents)
        
        # Calculate similarity scores
        scores = query_embeddings @ doc_embeddings.T
        
        # Apply normalization
        if normalize == "softmax":
            scores = torch.softmax(scores, dim=-1)
        elif normalize == "scale":
            scores = scores * 100
        elif normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")
            
        return scores

    def rerank(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        top_k: int = 5,
        normalize: str = "softmax"
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Rerank documents based on their semantic similarity to the query.
        
        Args:
            query: Query string or list of query strings
            documents: List of documents to rerank
            top_k: Number of top results to return per query
            normalize: Normalization method for scores
            
        Returns:
            List of dicts containing reranked documents and their scores.
            For single query: [{"document": str, "score": float}, ...]
            For multiple queries: [[{"document": str, "score": float}, ...], ...]
        """
        queries = [query] if isinstance(query, str) else query
        scores = self.calculate_scores(queries, documents, normalize=normalize)
        
        results = []
        for query_scores in scores:
            top_indices = torch.topk(query_scores, min(top_k, len(documents)), dim=0)
            query_results = [
                {
                    "document": documents[idx.item()],
                    "score": score.item()
                }
                for score, idx in zip(top_indices.values, top_indices.indices)
            ]
            results.append(query_results)
        
        return results[0] if isinstance(query, str) else results

    def get_reranked_documents(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        top_k: int = 5,
        normalize: str = "softmax"
    ) -> Union[List[str], List[List[str]]]:
        """
        Returns only the reranked documents without scores.
        
        Args:
            query: Query string or list of query strings
            documents: List of documents to rerank
            top_k: Number of top results to return per query
            normalize: Normalization method for scores
            
        Returns:
            For single query: List of reranked document strings
            For multiple queries: List of lists of reranked document strings
        """
        results = self.rerank(query, documents, top_k, normalize)
        return "\n".join([x['document'].strip() for x in results])
