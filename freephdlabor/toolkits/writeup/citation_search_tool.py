"""
CitationSearchTool - Search and format academic citations for LaTeX papers.

This tool provides comprehensive academic search capabilities including:
- arXiv paper search and metadata extraction
- Semantic Scholar integration for broader academic coverage
- BibTeX citation generation for LaTeX
- Citation formatting and validation
- Related work discovery and recommendation

Combines functionality from fetch_arxiv_papers and web search for academic sources.
"""

import json
import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from smolagents import Tool


class CitationSearchTool(Tool):
    name = "citation_search_tool"
    description = """
    Search for academic papers and generate properly formatted citations for LaTeX papers.
    
    This tool is essential for:
    - Finding relevant papers for your literature review
    - Generating BibTeX entries for LaTeX documents
    - Discovering related work in your research area
    - Validating and formatting academic citations
    - Building comprehensive bibliographies
    
    Use this tool when:
    - You need to cite papers in your LaTeX writeup
    - You want to find related work on a specific topic
    - You need properly formatted BibTeX entries
    - You're building a literature review section
    - You want to discover recent papers in your field
    
    The tool searches both arXiv and Semantic Scholar to provide comprehensive coverage
    of academic literature, with focus on computer science and machine learning papers.
    
    Input: Search query or paper title/author
    Output: Structured citations with BibTeX entries and metadata
    """
    
    inputs = {
        "search_query": {
            "type": "string", 
            "description": "Search query for papers (keywords, title, author, or topic)"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of papers to return (default: 10)",
            "nullable": True
        },
        "search_source": {
            "type": "string",
            "description": "Search database (default: 'both'):\n" +
                          "• 'arxiv': Search only arXiv preprints (faster, ML/CS focused, may miss published papers)\n" +
                          "• 'semantic_scholar': Search only Semantic Scholar (broader coverage, includes journals/conferences)\n" +
                          "• 'both': Search both databases for comprehensive results (recommended for literature reviews)",
            "nullable": True
        }
    }
    
    outputs = {
        "citations": {
            "type": "string",
            "description": "Structured citations with BibTeX entries and metadata"
        }
    }
    
    output_type = "string"

    def __init__(self):
        """Initialize CitationSearchTool."""
        super().__init__()
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
    def forward(self, search_query: str, max_results: int = 10, search_source: str = "both") -> str:
        """
        Search for academic papers and generate citations.
        
        Args:
            search_query: Search query for papers
            max_results: Maximum number of papers to return
            search_source: Source to search ('arxiv', 'semantic_scholar', or 'both')
            
        Returns:
            JSON string containing structured citations
        """
        try:
            citations = []
            
            if search_source in ["arxiv", "both"]:
                arxiv_results = self._search_arxiv(search_query, max(1, max_results // 2) if search_source == "both" else max_results)
                citations.extend(arxiv_results)

            if search_source in ["semantic_scholar", "both"]:
                ss_results = self._search_semantic_scholar(search_query, max(1, max_results // 2) if search_source == "both" else max_results)
                citations.extend(ss_results)
            
            # Remove duplicates based on title similarity
            citations = self._deduplicate_citations(citations)
            
            # Limit to max_results
            citations = citations[:max_results]
            
            # Generate BibTeX entries
            bibtex_entries = []
            for citation in citations:
                bibtex = self._generate_bibtex(citation)
                if bibtex:
                    bibtex_entries.append(bibtex)
            
            result = {
                "search_query": search_query,
                "search_source": search_source,
                "total_results": len(citations),
                "citations": citations,
                "bibtex_entries": bibtex_entries,
                "usage_instructions": {
                    "latex_integration": "Copy BibTeX entries to your .bib file and use \\cite{key} in LaTeX",
                    "citation_keys": [self._extract_citation_key(bibtex) for bibtex in bibtex_entries if bibtex]
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "error": f"Citation search failed: {str(e)}",
                "search_query": search_query,
                "citations": [],
                "bibtex_entries": []
            }
            return json.dumps(error_result, indent=2)
    
    def _search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search arXiv for papers."""
        try:
            # Add small delay to be respectful to arXiv API
            time.sleep(1)
            
            url = f"{self.arxiv_base_url}search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
            
            headers = {
                "User-Agent": "Academic-Citation-Tool/1.0 (research-tool)"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except Exception as e:
            print(f"Warning: arXiv search failed: {e}")
            return []
    
    def _search_semantic_scholar(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers with rate limiting."""
        try:
            # Add delay to respect rate limits (Semantic Scholar allows ~100 requests/5 minutes)
            time.sleep(2)  # 2 second delay between requests
            
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,year,abstract,citationCount,venue,externalIds,url"
            }
            
            headers = {
                "User-Agent": "Academic-Citation-Tool/1.0 (research-tool; contact@example.com)"
            }
            
            response = requests.get(
                self.semantic_scholar_base_url, 
                params=params, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 429:
                print("Warning: Semantic Scholar rate limit hit, waiting and retrying...")
                time.sleep(10)  # Wait 10 seconds before retry
                response = requests.get(
                    self.semantic_scholar_base_url, 
                    params=params, 
                    headers=headers,
                    timeout=30
                )
            
            response.raise_for_status()
            data = response.json()
            
            return self._parse_semantic_scholar_response(data)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Warning: Semantic Scholar rate limit exceeded. Try again later.")
                return []
            else:
                print(f"Warning: Semantic Scholar HTTP error: {e}")
                return []
        except Exception as e:
            print(f"Warning: Semantic Scholar search failed: {e}")
            return []
    
    def _parse_arxiv_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse arXiv API response XML."""
        try:
            root = ET.fromstring(response_text)
            papers = []
            
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                paper = {
                    "source": "arxiv",
                    "title": "",
                    "authors": [],
                    "year": "",
                    "abstract": "",
                    "url": "",
                    "arxiv_id": "",
                    "citation_count": 0,
                    "venue": "arXiv preprint"
                }
                
                # Title
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                if title_elem is not None:
                    paper["title"] = title_elem.text.strip()
                
                # Authors
                for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                    name_elem = author.find("{http://www.w3.org/2005/Atom}name")
                    if name_elem is not None:
                        paper["authors"].append(name_elem.text.strip())
                
                # Published date
                published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                if published_elem is not None:
                    paper["year"] = published_elem.text[:4]
                
                # Abstract
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                if summary_elem is not None:
                    paper["abstract"] = summary_elem.text.strip()
                
                # URL and arXiv ID
                id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                if id_elem is not None:
                    paper["url"] = id_elem.text
                    paper["arxiv_id"] = id_elem.text.split("/")[-1]
                
                if paper["title"]:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Warning: Failed to parse arXiv response: {e}")
            return []
    
    def _parse_semantic_scholar_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Semantic Scholar API response."""
        try:
            papers = []
            
            for item in data.get("data", []):
                paper = {
                    "source": "semantic_scholar",
                    "title": item.get("title", ""),
                    "authors": [author.get("name", "") for author in item.get("authors", [])],
                    "year": str(item.get("year", "")),
                    "abstract": item.get("abstract", ""),
                    "url": item.get("url", ""),
                    "arxiv_id": "",
                    "citation_count": item.get("citationCount", 0),
                    "venue": item.get("venue", "")
                }
                
                # Extract arXiv ID if present
                external_ids = item.get("externalIds", {})
                if external_ids and "ArXiv" in external_ids:
                    paper["arxiv_id"] = external_ids["ArXiv"]
                
                if paper["title"]:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Warning: Failed to parse Semantic Scholar response: {e}")
            return []
    
    def _deduplicate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate citations based on title similarity."""
        if not citations:
            return citations
        
        unique_citations = []
        seen_titles = set()
        
        for citation in citations:
            title = citation.get("title", "").lower().strip()
            # Simple deduplication - normalize title and check if similar exists
            normalized_title = re.sub(r'[^\w\s]', '', title).replace(' ', '')
            
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_citations.append(citation)
        
        return unique_citations
    
    def _generate_bibtex(self, citation: Dict[str, Any]) -> Optional[str]:
        """Generate BibTeX entry for a citation."""
        try:
            title = citation.get("title", "").strip()
            if not title:
                return None
            
            # Generate citation key
            first_author = citation.get("authors", ["Unknown"])[0] if citation.get("authors") else "Unknown"
            first_author_last = first_author.split()[-1] if " " in first_author else first_author
            year = citation.get("year", "")
            
            # Clean author name for key
            clean_author = re.sub(r'[^\w]', '', first_author_last.lower())
            citation_key = f"{clean_author}{year}"
            
            # Choose entry type
            if citation.get("arxiv_id"):
                entry_type = "misc"
                note_field = f"arXiv preprint arXiv:{citation['arxiv_id']}"
            elif citation.get("venue") and "conference" in citation.get("venue", "").lower():
                entry_type = "inproceedings"
                note_field = ""
            elif citation.get("venue") and "journal" in citation.get("venue", "").lower():
                entry_type = "article"
                note_field = ""
            else:
                entry_type = "misc"
                note_field = ""
            
            # Format authors
            authors = citation.get("authors", [])
            if authors:
                author_str = " and ".join(authors)
            else:
                author_str = "Unknown"
            
            # Build BibTeX entry
            bibtex_lines = [f"@{entry_type}{{{citation_key},"]
            bibtex_lines.append(f'  title = {{{title}}},')
            bibtex_lines.append(f'  author = {{{author_str}}},')
            
            if year:
                bibtex_lines.append(f'  year = {{{year}}},')
            
            if citation.get("venue"):
                if entry_type == "inproceedings":
                    bibtex_lines.append(f'  booktitle = {{{citation["venue"]}}},')
                elif entry_type == "article":
                    bibtex_lines.append(f'  journal = {{{citation["venue"]}}},')
            
            if note_field:
                bibtex_lines.append(f'  note = {{{note_field}}},')
            
            if citation.get("url"):
                bibtex_lines.append(f'  url = {{{citation["url"]}}},')
            
            bibtex_lines.append("}")
            
            return "\n".join(bibtex_lines)
            
        except Exception as e:
            print(f"Warning: Failed to generate BibTeX for citation: {e}")
            return None
    
    def _extract_citation_key(self, bibtex: str) -> Optional[str]:
        """Extract citation key from BibTeX entry."""
        try:
            match = re.search(r'@\w+\{([^,]+),', bibtex)
            return match.group(1) if match else None
        except:
            return None