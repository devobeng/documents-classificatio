import os

class Config:
    # Where the fine-tuned LegalBERT model + tokenizer live
    MODEL_DIR = os.getenv("MODEL_DIR", "./src/models")
    # Max request size to avoid memory abuse (10 MB default)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))
    # Tokenization / chunking
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 510))  # Leave room for [CLS] and [SEP] tokens
    STRIDE_TOKENS = int(os.getenv("STRIDE_TOKENS", 128))  # Increased stride for better coverage
    # OCR language (install the corresponding tesseract language pack if not "eng")
    OCR_LANG = os.getenv("OCR_LANG", "eng")
    # Environment feature flags
    ENABLE_OCR = os.getenv("ENABLE_OCR", "1") == "1"
