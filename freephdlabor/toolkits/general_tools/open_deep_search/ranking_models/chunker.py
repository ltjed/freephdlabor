from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    """A modular text chunking class that splits text into smaller, overlapping segments.
    
    This class provides a flexible way to break down large texts into smaller chunks
    while maintaining context through configurable overlap. It uses RecursiveCharacterTextSplitter
    from langchain under the hood.
    
    Attributes:
        chunk_size (int): The target size for each text chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        separators (List[str]): List of separators to use for splitting, in order of preference.
        length_function (callable): Function to measure text length (default: len).
    """

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: callable = len
    ):
        """Initialize the Chunker with specified parameters.
        
        Args:
            chunk_size (int, optional): Target size for each chunk. Defaults to 250.
            chunk_overlap (int, optional): Number of characters to overlap. Defaults to 50.
            separators (List[str], optional): Custom separators for splitting.
                Defaults to ["\n\n", "\n", " "].
            length_function (callable, optional): Function to measure text length.
                Defaults to len.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n"]
        self.length_function = length_function
        
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split a single text into chunks.
        
        Args:
            text (str): The input text to be split into chunks.
            
        Returns:
            List[str]: A list of text chunks.
        """
        return self.splitter.split_text(text)
    
    def split_texts(self, texts: List[str]) -> List[List[str]]:
        """Split multiple texts into chunks.
        
        Args:
            texts (List[str]): A list of input texts to be split into chunks.
            
        Returns:
            List[List[str]]: A list of lists, where each inner list contains
                the chunks for one input text.
        """
        return [self.split_text(text) for text in texts]
