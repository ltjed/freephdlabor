from typing import Optional, Literal
from smolagents import Tool
from .ods_agent import OpenDeepSearchAgent
import os
from dotenv import load_dotenv

load_dotenv()

class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng"] = "serper",
    ):
        super().__init__()
        self.search_model_name = model_name  # LiteLLM model name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.searxng_instance_url = os.getenv("SEARXNG_INSTANCE_URL")
        self.searxng_api_key = os.getenv("SEARXNG_API_KEY")
        self.setup()

    def forward(self, query: str):
        answer = self.search_tool.ask_sync(query, max_sources=2, pro_mode=True)
        return answer

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key
        )
