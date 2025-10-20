"""
Fetches and downloads papers from arXiv based on a search query.
This tool is taken from https://programmer.ie/post/deepresearch1/
"""


from smolagents import Tool

import os
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import re


SEARCH_QUERY= "agent"  # Replace with desired search term or topic
MAX_RESULTS= 50  # Adjust the number of papers you want to download
OUTPUT_FOLDER= "data"  # Folder to store downloaded papers
BASE_URL= "http://export.arxiv.org/api/query?"


class FetchArxivPapersTool(Tool):
    name = "fetch_arxiv_papers"
    description = """
    This is a tool will search arxiv based upn the query. I will return a configurable amount of papers ."""
    inputs = {
        "search_query": {
            "type": "string",
            "description": "The search query to use for finding papers on arXiv.",
        },
        "max_results": {
            "type": "integer",
            "description": "The maximum number of papers to return.",
            "nullable": True,
        }
    }
    output_type = "string"
    
    def __init__(self, working_dir=None):
        """
        Initialize the FetchArxivPapersTool.
        
        Args:
            working_dir: Optional working directory to save downloaded papers.
                        If not provided, uses the default "data" folder.
        """
        super().__init__()
        if working_dir:
            self.output_folder = os.path.join(working_dir, "ideation_agent", "downloaded_papers")
        else:
            self.output_folder = OUTPUT_FOLDER


    def sanitize_filename(self, title):
        """Sanitizes a string to be used as a filename."""
        # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
        return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")


    def get_filename_from_url(self, url):
        # Parse the URL to get the path component
        parsed_url = urlparse(url)
        # Get the base name from the URL's path
        filename = os.path.basename(parsed_url.path)
        return filename

    def compute_file_hash(self, file_path, algorithm="sha256"):
        """Compute the hash of a file using the specified algorithm."""
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as file:
            # Read the file in chunks of 8192 bytes
            while chunk := file.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def fetch_arxiv_papers(self, search_query, max_results=5):
        """Fetches metadata of papers from arXiv using the API."""
        url = f"{BASE_URL}search_query=all:{search_query}&start=0&max_results={max_results}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text


    def parse_paper_links(self, response_text):
        """Parses paper links and titles from arXiv API response XML."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response_text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            pdf_link = None
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"] + ".pdf"
                    break
            if pdf_link:
                title = self.get_filename_from_url(pdf_link)
                print(title)
                papers.append((title, pdf_link))
        return papers


    def download_paper(self, title, pdf_link, output_folder):
        """Downloads a single paper PDF."""
        # Create a safe filename
        safe_title = self.sanitize_filename(title)
        filename = os.path.join(output_folder, f"{safe_title}.pdf")
        response = requests.get(pdf_link, stream=True)
        response.raise_for_status()

        # Write the PDF to the specified folder
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {title}")


    def forward(self, search_query: str, max_results: int = 5):
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Fetch and parse papers
        print(f"Searching for papers on '{search_query}'...")
        response_text = self.fetch_arxiv_papers(search_query, max_results)
        papers = self.parse_paper_links(response_text)

        # Download each paper
        print(f"Found {len(papers)} papers. Starting download...")
        downloaded_papers = []
        for title, pdf_link in papers:
            try:
                safe_title = self.sanitize_filename(title)
                filename = os.path.join(self.output_folder, f"{safe_title}.pdf")
                self.download_paper(title, pdf_link, self.output_folder)
                downloaded_papers.append(filename)
                time.sleep(2)  # Pause to avoid hitting rate limits
            except Exception as e:
                print(f"Failed to download '{title}': {e}")
        
        return f"Download complete! Saved {len(downloaded_papers)} papers to the '{self.output_folder}' directory: {downloaded_papers}"