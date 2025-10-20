"""
Contains the ExtractionResult class for holding extraction operation results.
"""

from typing import Optional

class ExtractionResult:
    """Holds the results of an extraction operation"""
    def __init__(self, name: str, success: bool, content: Optional[str] = None, error: Optional[str] = None):
        self.name = name
        self.success = success
        self.content = content
        self.error = error
        self.raw_markdown_length = 0
        self.citations_markdown_length = 0

def print_extraction_result(result: ExtractionResult):
    """Utility function to print extraction results"""
    if result.success:
        print(f"\n=== {result.name} Results ===")
        print(f"Extracted Content: {result.content}")
        print(f"Raw Markdown Length: {result.raw_markdown_length}")
        print(f"Citations Markdown Length: {result.citations_markdown_length}")
    else:
        print(f"Error in {result.name}: {result.error}") 