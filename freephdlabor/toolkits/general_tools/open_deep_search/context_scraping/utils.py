import re
from typing import List, Tuple
import fasttext
from huggingface_hub import hf_hub_download
import wikipediaapi

# Load the model
model = fasttext.load_model(hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v2", "model.bin"))

def clean_markdown_links(text: str, min_quality_score: float = 0.2) -> Tuple[str, float]:
    """
    Clean markdown links and filter low-quality content.
    Returns tuple of (cleaned_text, quality_score)
    """
    # Split by double newlines to preserve paragraph structure
    paragraphs = text.split('\n\n')
    
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Preserve code blocks by checking if paragraph contains ``` tags
        if '```' in paragraph:
            cleaned_paragraphs.append(paragraph)
            continue
            
        lines = paragraph.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Keep headers regardless of length
            if re.match(r'^#{1,6}\s+', line):
                filtered_lines.append(line)
                continue
            
            # Skip common UI/navigation elements
            if re.match(r'^(Share|Trade|More|Buy|Sell|Download|Menu|Home|Back|Next|Previous|\d+\s*(BTC|USD|EUR|GBP)|\w{3}-\w{1,3}|Currency:.*|You (Buy|Spend|Receive)|≈|\d+\.\d+)', line, re.IGNORECASE):
                continue
                
            # Count words before removing markdown
            word_count = len(re.sub(r'\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>', '', line).split())
            
            # Increase minimum word threshold to 12
            if word_count < 12:
                # Check if line only contains markdown patterns or appears to be a currency/trading related line
                cleaned_line = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)|\[.*?\]\(.*?\)|!\[.*?\]\(.*?\)|<.*?>|\d+(\.\d+)?%?|\$\d+(\.\d+)?', '', line).strip()
                if not cleaned_line or len(cleaned_line.split()) < 8:  # If nothing substantial remains, skip this line
                    continue
            
            filtered_lines.append(line)
        
        # Only add paragraph if it has any lines left
        if filtered_lines:
            cleaned_paragraphs.append('\n'.join(filtered_lines))
    
    # Rejoin with double newlines
    cleaned_text = '\n\n'.join(cleaned_paragraphs)
    
    # Get quality score
    quality_score = predict_educational_value([cleaned_text])[0]
    
    return cleaned_text, quality_score

def filter_quality_content(text: str, min_quality_score: float = 0.2) -> str:
    """
    Filter content based on quality and returns concatenated quality content
    """
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Process each paragraph
    quality_content = []
    for paragraph in paragraphs:
        if not paragraph.strip():  # Skip empty paragraphs
            continue
            
        cleaned_text, quality_score = clean_markdown_links(paragraph, min_quality_score)
        if cleaned_text and quality_score >= min_quality_score:
            quality_content.append((cleaned_text, quality_score))
    
    # Debug print
    print(f"Found {len(quality_content)} quality paragraphs out of {len(paragraphs)} total")
    
    if quality_content:
        return "\n\n".join(text for text, _ in quality_content)
    return text  # Return original text if no quality content found

def replace_newlines(text: str) -> str:
    """Replace multiple newlines with a single space."""
    return re.sub("\n+", " ", text)

score_dict = {
    '__label__': 0, 
    '__label__Low': 0, 
    '__label__Mid': 1,
    '__label__High': 2
}

def predict_educational_value(text_list: List[str]) -> List[float]:
    """
    Predict educational value scores for a list of texts.
    Returns a list of scores between 0 and 2.
    """
    text_list = [replace_newlines(text) for text in text_list]
    pred = model.predict(text_list, k=-1)
    score_list = []
    for l, s in zip(*pred):
        score = 0
        for _l, _s in zip(l, s):
            score += score_dict[_l] * _s
        score_list.append(float(score))
    return score_list

def get_wikipedia_content(url: str) -> str | None:
    """
    Extract content from a Wikipedia URL.
    
    Args:
        url: Wikipedia URL to scrape
        
    Returns:
        str: Page content if found, None otherwise
    """
    wiki = wikipediaapi.Wikipedia(user_agent="opendeepsearch", language='en')
    
    # Extract the page title from URL (everything after /wiki/)
    try:
        title = url.split('/wiki/')[-1]
        page = wiki.page(title)
        if page.exists():
            return page.text
        return None
    except Exception:
        return None

# Patterns
SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"
IFRAME_PATTERN = r"<[ ]*iframe.*?\/[ ]*iframe[ ]*>"
NOSCRIPT_PATTERN = r"<[ ]*noscript.*?\/[ ]*noscript[ ]*>"
HEADER_PATTERN = r"<[ ]*header.*?\/[ ]*header[ ]*>"
FOOTER_PATTERN = r"<[ ]*footer.*?\/[ ]*footer[ ]*>"
NAV_PATTERN = r"<[ ]*nav.*?\/[ ]*nav[ ]*>"
FORM_PATTERN = r"<[ ]*form.*?\/[ ]*form[ ]*>"


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    """Clean HTML content by removing various elements."""
    patterns = [
        SCRIPT_PATTERN,
        STYLE_PATTERN,
        META_PATTERN,
        COMMENT_PATTERN,
        LINK_PATTERN,
        IFRAME_PATTERN,
        NOSCRIPT_PATTERN,
        HEADER_PATTERN,
        FOOTER_PATTERN,
        NAV_PATTERN,
        FORM_PATTERN
    ]
    
    for pattern in patterns:
        html = re.sub(pattern, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
        
    # Remove empty lines and excessive whitespace
    html = re.sub(r'\n\s*\n', '\n', html)
    html = re.sub(r'\s+', ' ', html)
    
    return html.strip()

JSON_SCHEMA = """
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string"
    },
    "author": {
      "type": "string"
    },
    "date": {
      "type": "string"
    },
    "content": {
      "type": "string"
    }
  },
  "required": ["title", "author", "date", "content"]
}
"""