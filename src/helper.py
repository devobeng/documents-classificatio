import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import string
import pdfplumber

# Keep hyphen and slash so dates like 2023-04-26 and 12/05/2024 remain intact
_PUNCT_TO_STRIP = string.punctuation.translate({ord('-'): None, ord('/'): None})


def clean_contract_text(text: str) -> str:
    """Clean contract/legal text while preserving dates, numbers, and structure."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # remove square-bracket notes
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove urls
    text = re.sub(r'<.*?>+', '', text)  # remove html
    # remove only junk punctuation; keep hyphen & slash for dates; keep dots/colons/semicolons/() for structure
    text = text.translate(str.maketrans('', '', _PUNCT_TO_STRIP.replace('.', '').replace(':', '').replace(';', '').replace('(', '').replace(')', '')))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_path):
    """Extract text from PDF using pdfplumber"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

def extract_text_from_pdf_ocr(file_path):
    """Extract text from scanned PDFs using OCR"""
    text = ""
    # Convert PDF pages to images
    pages = convert_from_path(file_path)
    for page in pages:
        # Run OCR on each page
        text += pytesseract.image_to_string(page) + " "
    return text


def extract_text_smart(file_path):
    # Try normal extraction
    text = extract_text_from_pdf(file_path)
    if not text.strip():  # If empty (likely scanned PDF)
        print(" No text found, using OCR...")
        text = extract_text_from_pdf_ocr(file_path)
    return text

def format_snippet(text, max_len=500):
    """Format extracted text snippet for JSON response"""
    # Collapse multiple spaces/newlines into single spaces
    snippet = re.sub(r'\s+', ' ', text).strip()
    # Limit length
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet

def sniff_is_pdf(filename: str, mimetype: str) -> bool:
    """Check if file is likely a PDF based on filename and mimetype"""
    if not filename:
        return False
    
    filename_lower = filename.lower()
    return (filename_lower.endswith('.pdf') or 
            mimetype == 'application/pdf' or 
            mimetype == 'application/octet-stream')


