"""
Contains the StrategyFactory class for creating various extraction strategies.
"""

import os
from typing import Optional

from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
    JsonXPathExtractionStrategy,
    NoExtractionStrategy,
    CosineStrategy,
)

class StrategyFactory:
    """Factory for creating extraction strategies"""
    @staticmethod
    def create_llm_strategy(
        input_format: str = "markdown",
        instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
    ) -> LLMExtractionStrategy:
        return LLMExtractionStrategy(
            input_format=input_format,
            provider="openrouter/google/gemini-2.0-flash-lite-001",  # Uses LiteLLM as provider
            api_token=os.getenv("OPENROUTER_API_KEY"),
            instruction=instruction
        )

    @staticmethod
    def create_css_strategy() -> JsonCssExtractionStrategy:
        schema = {
            "baseSelector": ".product",
            "fields": [
                {"name": "title", "selector": "h1.product-title", "type": "text"},
                {"name": "price", "selector": ".price", "type": "text"},
                {"name": "description", "selector": ".description", "type": "text"},
            ],
        }
        return JsonCssExtractionStrategy(schema=schema)

    @staticmethod
    def create_xpath_strategy() -> JsonXPathExtractionStrategy:
        schema = {
            "baseSelector": "//div[@class='product']",
            "fields": [
                {"name": "title", "selector": ".//h1[@class='product-title']/text()", "type": "text"},
                {"name": "price", "selector": ".//span[@class='price']/text()", "type": "text"},
                {"name": "description", "selector": ".//div[@class='description']/text()", "type": "text"},
            ],
        }
        return JsonXPathExtractionStrategy(schema=schema)

    @staticmethod
    def create_no_extraction_strategy() -> NoExtractionStrategy:
        return NoExtractionStrategy()

    @staticmethod
    def create_cosine_strategy(
        semantic_filter: Optional[str] = None,
        word_count_threshold: int = 10,
        max_dist: float = 0.2,
        sim_threshold: float = 0.3,
        debug: bool = False
    ) -> CosineStrategy:
        return CosineStrategy(
            semantic_filter=semantic_filter,
            word_count_threshold=word_count_threshold,
            max_dist=max_dist,
            sim_threshold=sim_threshold,
            verbose=debug
        ) 