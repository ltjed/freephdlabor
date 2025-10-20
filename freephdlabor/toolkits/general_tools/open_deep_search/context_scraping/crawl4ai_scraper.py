"""
Modular web scraping implementation using Crawl4AI.
Supports multiple extraction strategies including LLM, CSS, and XPath.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from freephdlabor.toolkits.general_tools.open_deep_search.context_scraping.extraction_result import (
    ExtractionResult,
    print_extraction_result,
)
from freephdlabor.toolkits.general_tools.open_deep_search.context_scraping.basic_web_scraper import (
    ExtractionConfig,
)
from freephdlabor.toolkits.general_tools.open_deep_search.context_scraping.strategy_factory import (
    StrategyFactory,
)


class WebScraper:
    """Unified scraper that encapsulates all extraction strategies and configuration"""

    def __init__(
        self,
        browser_config: Optional[BrowserConfig] = None,
        strategies: List[str] = ["no_extraction"],
        llm_instruction: str = "Extract relevant content from the provided text, only return the text, no markdown formatting, remove all footnotes, citations, and other metadata and only keep the main content",
        user_query: Optional[str] = None,
        debug: bool = False,
        filter_content: bool = False,
    ):
        self.browser_config = browser_config or BrowserConfig(
            headless=True, verbose=True
        )
        self.debug = debug
        self.factory = StrategyFactory()
        self.strategies = strategies or [
            "markdown_llm",
            "html_llm",
            "fit_markdown_llm",
            "css",
            "xpath",
            "no_extraction",
            "cosine",
        ]
        self.llm_instruction = llm_instruction
        self.user_query = user_query
        self.filter_content = filter_content

        # Validate strategies
        valid_strategies = {
            "markdown_llm",
            "html_llm",
            "fit_markdown_llm",
            "css",
            "xpath",
            "no_extraction",
            "cosine",
        }
        invalid_strategies = set(self.strategies) - valid_strategies
        if invalid_strategies:
            raise ValueError(f"Invalid strategies: {invalid_strategies}")

        # Initialize strategy map
        self.strategy_map = {
            "markdown_llm": lambda: self.factory.create_llm_strategy(
                "markdown", self.llm_instruction
            ),
            "html_llm": lambda: self.factory.create_llm_strategy(
                "html", self.llm_instruction
            ),
            "fit_markdown_llm": lambda: self.factory.create_llm_strategy(
                "fit_markdown", self.llm_instruction
            ),
            "css": self.factory.create_css_strategy,
            "xpath": self.factory.create_xpath_strategy,
            "no_extraction": self.factory.create_no_extraction_strategy,
            "cosine": lambda: self.factory.create_cosine_strategy(debug=self.debug),
        }

    def _create_crawler_config(self) -> CrawlerRunConfig:
        """Creates default crawler configuration"""
        content_filter = (
            PruningContentFilter(user_query=self.user_query)
            if self.user_query
            else PruningContentFilter()
        )
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
        )

    async def scrape(self, url: str) -> Dict[str, ExtractionResult]:
        """
        Scrape URL using configured strategies

        Args:
            url: Target URL to scrape
        """
        # Handle Wikipedia URLs
        if "wikipedia.org/wiki/" in url:
            from general_tools.open_deep_search.context_scraping.utils import (
                get_wikipedia_content,
            )

            try:
                content = get_wikipedia_content(url)
                # Create same result for all strategies since we're using Wikipedia content
                return {
                    strategy_name: ExtractionResult(
                        name=strategy_name, success=True, content=content
                    )
                    for strategy_name in self.strategies
                }
            except Exception as e:
                if self.debug:
                    print(f"Debug: Wikipedia extraction failed: {str(e)}")
                # If Wikipedia extraction fails, fall through to normal scraping

        # Normal scraping for non-Wikipedia URLs or if Wikipedia extraction failed
        results = {}
        for strategy_name in self.strategies:
            config = ExtractionConfig(
                name=strategy_name, strategy=self.strategy_map[strategy_name]()
            )
            result = await self.extract(config, url)
            results[strategy_name] = result

        return results

    async def scrape_many(
        self, urls: List[str]
    ) -> Dict[str, Dict[str, ExtractionResult]]:
        """
        Scrape multiple URLs using configured strategies in parallel

        Args:
            urls: List of target URLs to scrape

        Returns:
            Dictionary mapping URLs to their extraction results
        """
        # Create tasks for all URLs
        tasks = [self.scrape(url) for url in urls]
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks)

        # Build results dictionary
        results = {}
        for url, result in zip(urls, results_list):
            results[url] = result

        return results

    async def extract(
        self, extraction_config: ExtractionConfig, url: str
    ) -> ExtractionResult:
        """Internal method to perform extraction using specified strategy"""
        try:
            config = self._create_crawler_config()
            config.extraction_strategy = extraction_config.strategy

            if self.debug:
                print(
                    f"\nDebug: Attempting extraction with strategy: {extraction_config.name}"
                )
                print(f"Debug: URL: {url}")
                print(f"Debug: Strategy config: {config.extraction_strategy}")
                if self.user_query:
                    print(f"Debug: User query: {self.user_query}")

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                if isinstance(url, list):
                    result = await crawler.arun_many(urls=url, config=config)
                else:
                    result = await crawler.arun(url=url, config=config)

            if self.debug:
                print(f"Debug: Raw result attributes: {dir(result)}")
                print(f"Debug: Raw result: {result.__dict__}")

            # Handle different result formats based on strategy
            content = None
            if result.success:
                if extraction_config.name in ["no_extraction", "cosine"]:
                    # For strategies that return a list of dictionaries
                    if hasattr(result, "markdown_v2"):
                        content = result.markdown_v2.raw_markdown
                    elif hasattr(result, "raw_html"):
                        content = result.raw_html
                    elif (
                        hasattr(result, "extracted_content")
                        and result.extracted_content
                    ):
                        if isinstance(result.extracted_content, list):
                            content = "\n".join(
                                item.get("content", "")
                                for item in result.extracted_content
                            )
                        else:
                            content = result.extracted_content

                    if self.filter_content and content:
                        from general_tools.open_deep_search.context_scraping.utils import (
                            filter_quality_content,
                        )

                        content = filter_quality_content(content)
                else:
                    content = result.extracted_content
                    if self.filter_content and content:
                        from general_tools.open_deep_search.context_scraping.utils import (
                            filter_quality_content,
                        )

                        content = filter_quality_content(content)

            if self.debug:
                print(f"Debug: Processed content: {content[:200] if content else None}")

            extraction_result = ExtractionResult(
                name=extraction_config.name,
                success=result.success,
                content=content,
                error=getattr(result, "error", None),  # Capture error if available
            )

            if result.success:
                extraction_result.raw_markdown_length = len(
                    result.markdown_v2.raw_markdown
                )
                extraction_result.citations_markdown_length = len(
                    result.markdown_v2.markdown_with_citations
                )
            elif self.debug:
                print(f"Debug: Final extraction result: {extraction_result.__dict__}")

            return extraction_result

        except Exception as e:
            if self.debug:
                import traceback

                print(f"Debug: Exception occurred during extraction:")
                print(traceback.format_exc())

            return ExtractionResult(
                name=extraction_config.name, success=False, error=str(e)
            )


async def main():
    # Example usage with single URL
    single_url = "https://example.com/product-page"
    scraper = WebScraper(debug=True)
    results = await scraper.scrape(single_url)

    # Print single URL results
    for result in results.values():
        print_extraction_result(result)

    # Example usage with multiple URLs
    urls = ["https://example.com", "https://python.org", "https://github.com"]

    multi_results = await scraper.scrape_many(urls)

    # Print multiple URL results
    for url, url_results in multi_results.items():
        print(f"\nResults for {url}:")
        for result in url_results.values():
            print_extraction_result(result)


if __name__ == "__main__":
    asyncio.run(main())
