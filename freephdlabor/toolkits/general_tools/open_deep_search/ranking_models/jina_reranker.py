import requests
import torch
from typing import List, Optional
from dotenv import load_dotenv
import os
from .base_reranker import BaseSemanticSearcher

class JinaReranker(BaseSemanticSearcher):
    """
    Semantic searcher implementation using Jina AI's embedding API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "jina-embeddings-v3"):
        """
        Initialize the Jina reranker.
        
        Args:
            api_key: Jina AI API key. If None, will load from environment variable JINA_API_KEY
            model: Model name to use (default: "jina-embeddings-v3")
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('JINA_API_KEY')
            if not api_key:
                raise ValueError("No API key provided and JINA_API_KEY not found in environment variables")
        
        self.api_url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.model = model

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts using Jina AI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            torch.Tensor containing the embeddings
        """
        data = {
            "model": self.model,
            "task": "text-matching",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": texts
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=data)
            response.raise_for_status()  # Raise exception for non-200 status codes
            
            # Extract embeddings from response
            embeddings_data = [item["embedding"] for item in response.json()["data"]]
            
            # Convert to torch tensor
            embeddings = torch.tensor(embeddings_data)
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Jina AI API: {str(e)}")
