import json
import os
import time
from typing import List, Dict, Union, Optional

import backoff
import requests
from smolagents import Tool


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def _search_semantic_scholar(query: str, result_limit: int, s2_api_key: str) -> Union[None, List[Dict]]:
    """Helper function to search Semantic Scholar API"""
    if not query:
        return None
    
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": s2_api_key},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


class PaperSearchTool(Tool):
    name = "PaperSearchTool"
    description = "Searches Semantic Scholar for academic papers based on a query. Returns a list of papers with titles, authors, abstracts, and other metadata."
    
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query for academic papers"
        },
        "result_limit": {
            "type": "integer",
            "description": "Maximum number of papers to return",
            "nullable": True
        },
        "fields_of_study": {
            "type": "string",
            "description": "Comma-separated fields of study to filter by",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.s2_api_key = os.getenv("S2_API_KEY")
        if not self.s2_api_key:
            print("Warning: S2_API_KEY environment variable not set. Paper search may not work.")

    def forward(self, query: str, result_limit: int = 10, fields_of_study: str = None) -> str:
        """
        Search for academic papers on Semantic Scholar.
        
        Args:
            query: Search query for papers
            result_limit: Maximum number of papers to return (default: 10)
            fields_of_study: Comma-separated fields of study to filter by
            
        Returns:
            JSON string containing paper information or error message
        """
        try:
            if not self.s2_api_key:
                return json.dumps({"error": "S2_API_KEY not configured"})
            
            papers = _search_semantic_scholar(query, result_limit, self.s2_api_key)
            
            if papers is None:
                return json.dumps({"message": "No papers found", "papers": []})
            
            # Format papers for return
            formatted_papers = []
            for paper in papers:
                formatted_paper = {
                    "title": paper.get("title", ""),
                    "authors": [author.get("name", "") for author in paper.get("authors", [])],
                    "venue": paper.get("venue", ""),
                    "year": paper.get("year", ""),
                    "abstract": paper.get("abstract", ""),
                    "citationCount": paper.get("citationCount", 0)
                }
                formatted_papers.append(formatted_paper)
            
            return json.dumps({
                "message": f"Found {len(formatted_papers)} papers",
                "papers": formatted_papers
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error searching papers: {str(e)}"}) 