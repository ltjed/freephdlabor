"""
Enhanced web scraping implementation using Crawl4AI and vLLM.
Supports multiple extraction strategies with LLM-powered content processing.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from vllm import LLM, SamplingParams

from freephdlabor.toolkits.general_tools.open_deep_search.context_scraping.extraction_result import ExtractionResult
from freephdlabor.toolkits.general_tools.open_deep_search.context_scraping.utils import clean_html, get_wikipedia_content

@dataclass
class LLMConfig:
    """Configuration for LLM-based extraction"""
    model_name: str = 'jinaai/ReaderLM-v2'
    max_model_len: int = 512_000
    temperature: float = 0.0
    top_k: int = 1
    presence_penalty: float = 0.25
    frequency_penalty: float = 0.25
    repetition_penalty: float = 1.13
    max_tokens: int = 16_384

# DEFAULT_SCHEMA = """
# {
#   "type": "object",
#   "properties": {
#     "title": {
#       "type": "string"
#     },
#     "author": {
#       "type": "string"
#     },
#     "date": {
#       "type": "string"
#     },
#     "content": {
#       "type": "string"
#     }
#   },
#   "required": ["title", "author", "date", "content"]
# }
# """

class FastWebScraper:
    """Enhanced scraper with LLM-powered extraction and multiple strategies"""
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ):
        self.debug = debug
        self.browser_config = browser_config or BrowserConfig(headless=True, verbose=debug)
        self.llm_config = llm_config or LLMConfig()
        self.json_schema = None #json_schema or json.loads(DEFAULT_SCHEMA)
        
        # Initialize LLM
        self.sampling_params = SamplingParams(
            temperature=self.llm_config.temperature,
            top_k=self.llm_config.top_k,
            presence_penalty=self.llm_config.presence_penalty,
            repetition_penalty=self.llm_config.repetition_penalty,
            max_tokens=self.llm_config.max_tokens,
            frequency_penalty=self.llm_config.frequency_penalty
        )
        
        self.llm = LLM(
            model=self.llm_config.model_name,
            max_model_len=self.llm_config.max_model_len,
            dtype='float16'
        )
        
        self.tokenizer = self.llm.get_tokenizer()

    def _create_prompt(self, text: str, instruction: Optional[str] = None) -> str:
        """Create a prompt for the LLM"""
        if not instruction:
            instruction = "Extract the main content and convert to structured format."
        
        if self.json_schema:
            instruction = "Extract information according to the schema and return JSON."
            prompt = f"{instruction}\n```html\n{text}\n```\nSchema:```json\n{json.dumps(self.json_schema, indent=2)}\n```"
        else:
            prompt = f"{instruction}\n```html\n{text}\n```"

        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def _extract_content(self, html: str, instruction: Optional[str] = None) -> str:
        """Extract content using LLM"""
        cleaned_html = clean_html(html, clean_svg=True, clean_base64=True)
        prompt = self._create_prompt(cleaned_html, instruction)
        
        outputs = self.llm.generate(prompt, self.sampling_params)
        raw_text = outputs[0].outputs[0].text
        return self._parse_llm_output(raw_text)

    def _parse_llm_output(self, text: str) -> str:
        """
        Parse LLM output, handling both single dictionaries and lists of dictionaries.
        Returns the content field from the most appropriate dictionary.
        """
        try:
            # Strip any markdown code block markers
            text = text.strip()
            if text.startswith('```') and text.endswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            
            data = json.loads(text.strip())
            
            if isinstance(data, dict):
                return data.get('content', '')
            
            if isinstance(data, list):
                # First try to find a dictionary with non-empty content
                for item in data:
                    if isinstance(item, dict) and item.get('content'):
                        return item['content']
                
                # If no content found, return content from last item or empty string
                last_item = data[-1]
                return last_item.get('content', '') if isinstance(last_item, dict) else ''
            
            return ''
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return the original text
            return text.strip()
        except Exception:
            return ''

    async def scrape(self, url: str, instruction: Optional[str] = None) -> ExtractionResult:
        """
        Scrape and process content from a URL
        
        Args:
            url: Target URL to scrape
            instruction: Optional custom instruction for the LLM
        """
        try:
            if self.debug:
                print(f"Debug: Processing URL: {url}")

            # Handle Wikipedia URLs
            if 'wikipedia.org/wiki/' in url:
                try:
                    content = get_wikipedia_content(url)
                    return ExtractionResult(
                        name="llm_extraction",
                        success=True,
                        content=content
                    )
                except Exception as e:
                    if self.debug:
                        print(f"Debug: Wikipedia extraction failed: {str(e)}")
                    # If Wikipedia extraction fails, fall through to normal scraping

            # Fetch HTML
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(url=url, config=CrawlerRunConfig())
                
            if not result.success:
                return ExtractionResult(
                    name="llm_extraction",
                    success=False,
                    error="Failed to fetch HTML"
                )

            # Process with LLM
            content = await self._extract_content(result.html, instruction)
            
            return ExtractionResult(
                name="llm_extraction",
                success=True,
                content=content
            )

        except Exception as e:
            if self.debug:
                import traceback
                print(f"Debug: Exception during scraping:")
                print(traceback.format_exc())
            
            return ExtractionResult(
                name="llm_extraction",
                success=False,
                error=str(e)
            )

    async def scrape_many(self, urls: List[str], instruction: Optional[str] = None) -> Dict[str, ExtractionResult]:
        """
        Scrape multiple URLs
        
        Args:
            urls: List of target URLs
            instruction: Optional custom instruction for the LLM
        """
        results = {}
        for url in urls:
            results[url] = await self.scrape(url, instruction)
        return results
