"""
Contains the BasicWebScraper class for basic web scraping functionality.
"""

from dataclasses import dataclass
from typing import Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from freephdlabor.toolkits.general_tools.open_deep_search.context_scraping.extraction_result import ExtractionResult
from crawl4ai.extraction_strategy import ExtractionStrategy

@dataclass
class ExtractionConfig:
    """Configuration for extraction strategies"""
    name: str
    strategy: ExtractionStrategy 

class BasicWebScraper:
    """Basic web scraper implementation"""
    def __init__(self, browser_config: Optional[BrowserConfig] = None):
        self.browser_config = browser_config or BrowserConfig(headless=True, verbose=True)
        
    def _create_crawler_config(self) -> CrawlerRunConfig:
        """Creates default crawler configuration"""
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            )
        )

    async def extract(self, extraction_config: ExtractionConfig, url: str) -> ExtractionResult:
        """Performs extraction using specified strategy"""
        try:
            config = self._create_crawler_config()
            config.extraction_strategy = extraction_config.strategy

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=config)

            extraction_result = ExtractionResult(
                name=extraction_config.name,
                success=result.success,
                content=result.extracted_content
            )
            
            if result.success:
                extraction_result.raw_markdown_length = len(result.markdown_v2.raw_markdown)
                extraction_result.citations_markdown_length = len(result.markdown_v2.markdown_with_citations)

            return extraction_result

        except Exception as e:
            return ExtractionResult(
                name=extraction_config.name,
                success=False,
                error=str(e)
            ) 