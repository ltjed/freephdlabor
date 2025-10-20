import torch
import requests
import json
from typing import List
from freephdlabor.toolkits.general_tools.open_deep_search.ranking_models.base_reranker import (
    BaseSemanticSearcher,
)


class InfinitySemanticSearcher(BaseSemanticSearcher):
    """
    A semantic reranking model that uses the Infinity Embedding API for text embeddings.

    This class provides methods to rerank documents based on their semantic similarity
    to queries using embeddings from the Infinity API. The API endpoint expects to receive
    text inputs and returns high-dimensional embeddings that capture semantic meaning.

    The default model used is 'Alibaba-NLP/gte-Qwen2-7B-instruct', but other models
    available through the Infinity API can be specified.

    Attributes:
        embedding_endpoint (str): URL of the Infinity Embedding API endpoint
        model_name (str): Name of the embedding model to use

    Example:
        ```python
        reranker = SemanticSearch(
            embedding_endpoint="http://localhost:7997/embeddings",
            model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
        )

        documents = [
            "Munich is in Germany.",
            "The sky is blue."
        ]

        results = reranker.rerank(
            query="What color is the sky?",
            documents=documents,
            top_k=1
        )
        ```
    """

    def __init__(
        self,
        embedding_endpoint: str = "http://localhost:7997/embeddings",
        model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_prefix: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    ):
        """
        Initialize the semantic search engine with Infinity Embedding API settings.

        Args:
            embedding_endpoint: URL of the Infinity Embedding API endpoint
            model_name: Name of the embedding model available in Infinity API
            instruction_prefix: Prefix to add to queries for better search relevance
        """
        self.embedding_endpoint = embedding_endpoint
        self.model_name = model_name
        self.instruction_prefix = instruction_prefix

    def _get_embeddings(
        self, texts: List[str], embedding_type: str = "query"
    ) -> torch.Tensor:
        """
        Get embeddings for a list of texts using the Infinity API.
        """
        MAX_TEXTS = 2048
        if len(texts) > MAX_TEXTS:
            import warnings

            warnings.warn(
                f"Number of texts ({len(texts)}) exceeds maximum of {MAX_TEXTS}. List will be truncated."
            )
            texts = texts[:MAX_TEXTS]

        # Format queries with instruction prefix
        formatted_texts = [
            self.instruction_prefix + text if embedding_type == "query" else text
            for text in texts
        ]

        response = requests.post(
            self.embedding_endpoint,
            json={"model": self.model_name, "input": formatted_texts},
        )

        content_str = response.content.decode("utf-8")
        content_json = json.loads(content_str)
        return torch.tensor([item["embedding"] for item in content_json["data"]])
