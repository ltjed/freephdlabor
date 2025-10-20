from smolagents import Tool
from smolagents.models import Model


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
Analyze complex documents and extract information. Use this for research papers (PDFs), Word docs, presentations, or when you need to ask questions about file content.
Handles: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), HTML, audio files (transcription), and plain text.
NOT for: Images, or simple workspace file reading (use see_file for that).
Can answer questions about the content using AI analysis."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model = None, text_limit: int = 100000, working_dir: str = None):
        super().__init__()
        from ...model_utils import get_raw_model
        self.model = get_raw_model(model)
        self.text_limit = text_limit
        self.working_dir = working_dir
        # Set chunk size to stay within rate limits - aim for ~1500 tokens per chunk
        self.chunk_size = 6000  # chars, roughly 1500 tokens
        # Configurable delay based on rate limits (60 / RPM to be safe)
        import os
        rpm_limit = int(os.environ.get("ANTHROPIC_RPM_LIMIT", "5"))
        self.api_delay = max(60 / rpm_limit, 10)  # At least 10 seconds between calls
        from ..text_web_browser.mdconvert import MarkdownConverter

        self.md_converter = MarkdownConverter()

    def _safe_path(self, path: str) -> str:
        """Convert path to absolute path, resolving relative paths with working_dir if provided."""
        if not self.working_dir:
            # No working directory - use path as-is (supports absolute paths and paths relative to current dir) 
            return path
            
        import os
        # If path is already absolute, use it directly
        if os.path.isabs(path):
            return path
        else:
            # Relative path - join with working_dir to create absolute path
            return os.path.abspath(os.path.join(self.working_dir, path))

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that fit within token limits."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Try to find a natural break point (paragraph, sentence, etc.)
            end_pos = current_pos + self.chunk_size
            
            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break
            
            # Look for natural break points in the last 500 chars
            chunk_end = text[current_pos:end_pos]
            
            # Try to break on paragraph boundaries first
            last_paragraph = chunk_end.rfind('\n\n')
            if last_paragraph > self.chunk_size * 0.5:  # At least 50% of chunk size
                end_pos = current_pos + last_paragraph
            else:
                # Fall back to sentence boundaries
                last_sentence = chunk_end.rfind('. ')
                if last_sentence > self.chunk_size * 0.5:
                    end_pos = current_pos + last_sentence + 1
                else:
                    # Fall back to word boundaries
                    last_space = chunk_end.rfind(' ')
                    if last_space > self.chunk_size * 0.5:
                        end_pos = current_pos + last_space
            
            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos
        
        return chunks

    def _summarize_chunk(self, chunk: str, question: str, chunk_num: int, total_chunks: int) -> str:
        """Summarize a single chunk with focus on the research question."""
        import time
        
        messages = [
            {
                "role": "system",
                "content": f"You are analyzing part {chunk_num} of {total_chunks} of a research paper. "
                          f"Extract key information relevant to this question: {question}\n"
                          f"Focus on: methods, findings, novelty, limitations, and connections to the question."
            },
            {
                "role": "user",
                "content": f"Document section:\n\n{chunk}\n\n"
                          f"Provide a concise summary (max 200 words) highlighting information relevant to: {question}"
            },
        ]
        
        # Add delay between API calls to avoid rate limiting
        if chunk_num > 1:
            time.sleep(self.api_delay)  # Configurable delay based on rate limits
        
        try:
            return self.model(messages)
        except Exception as e:
            return f"Error processing chunk {chunk_num}: {str(e)}"

    def forward_initial_exam_mode(self, file_path, question):
        safe_file_path = self._safe_path(file_path)
        result = self.md_converter.convert(safe_file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": "system",
                "content": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
            },
            {
                "role": "user",
                "content": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
            },
        ]
        return self.model(messages)

    def forward(self, file_path, question: str | None = None) -> str:
        import time

        safe_file_path = self._safe_path(file_path)
        result = self.md_converter.convert(safe_file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        # If text is small enough, use original approach
        if len(result.text_content) <= self.chunk_size:
            messages = [
                {
                    "role": "system",
                    "content": "You will have to write a short caption for this file, then answer this question:"
                            + question,
                },
                {
                    "role": "user",
                    "content": "Here is the complete file:\n### "
                            + str(result.title)
                            + "\n\n"
                            + result.text_content[: self.text_limit],
                },
                {
                    "role": "user",
                    "content": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                            + question,
                },
            ]
            return self.model(messages)

        # For large documents, use chunking approach
        print(f"Processing large document ({len(result.text_content)} chars) in chunks...")
        chunks = self._chunk_text(result.text_content[:self.text_limit])
        
        # Process each chunk and collect summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            summary = self._summarize_chunk(chunk, question, i, len(chunks))
            chunk_summaries.append(f"Section {i}: {summary}")
        
        # Combine all summaries into final analysis
        combined_summary = "\n\n".join(chunk_summaries)
        
        # Add delay before final synthesis
        time.sleep(self.api_delay)
        
        # Final synthesis of all chunks
        final_messages = [
            {
                "role": "system",
                "content": f"You are synthesizing analysis from {len(chunks)} sections of a research paper titled: {result.title}\n"
                          f"Provide a comprehensive answer to: {question}"
            },
            {
                "role": "user",
                "content": f"Here are the section summaries:\n\n{combined_summary}\n\n"
                          f"Now provide a comprehensive answer using these headings:\n"
                          f"1. Short answer\n2. Extremely detailed answer\n3. Additional Context on the document and question asked\n\n"
                          f"Question: {question}"
            },
        ]
        
        try:
            return self.model(final_messages)
        except Exception as e:
            return f"Error in final synthesis: {str(e)}\n\nChunk summaries:\n{combined_summary}"
