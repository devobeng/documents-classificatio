# Legal Contract Classification System

A comprehensive machine learning system for automatically classifying legal contracts and agreements using both traditional machine learning (SVM) and deep learning (LegalBERT) approaches.

## 🎯 Project Overview

This system can classify legal documents into 10 different contract types:

- License Agreement
- Employment Agreement
- Non-Disclosure Agreement
- Vendor Agreement
- Loan Agreement
- Partnership Agreement
- Consulting Agreement
- Service Level Agreement
- Franchise Agreement
- Lease Agreement

## 🏗️ Architecture

The system consists of two classification approaches:

1. **Traditional ML (SVM + TF-IDF)**: Fast, lightweight classification using Support Vector Machine with TF-IDF features
2. **Deep Learning (LegalBERT)**: Advanced transformer-based classification using fine-tuned BERT model

## 📁 Project Structure

```
assigment/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
├── test_model.py                  # Model testing script
├── data/
│   └── scontracts.csv            # Raw contract dataset
├── research/
│   ├── legalBert.ipynb           # BERT model training notebook
│   ├── trials.ipynb              # SVM model training notebook
│   ├── contracts_cleaned.csv     # Cleaned training data
│   ├── svm_contract_classifier.pkl # Trained SVM model
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer
└── src/
    ├── __init__.py
    ├── config.py                 # Configuration settings
    ├── model.py                  # LegalBERT classifier implementation
    ├── helper.py                 # Text processing utilities
    ├── models/                   # Fine-tuned BERT model files
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer.json
    │   └── vocab.txt
    └── utils/
        ├── __init__.py
        └── text_processing.py    # Text preprocessing utilities
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd assigment
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

The server will start on `http://localhost:8080`

## 🔧 Configuration

Key configuration settings in `src/config.py`:

```python
class Config:
    MODEL_DIR = "./src/models"              # BERT model directory
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024   # Max file upload size (10MB)
    MAX_TOKENS = 510                        # Max tokens per chunk
    STRIDE_TOKENS = 128                     # Token overlap between chunks
    OCR_LANG = "eng"                        # OCR language
    ENABLE_OCR = True                       # Enable OCR for scanned PDFs
```

## 📡 API Endpoints

### 1. Health Check

```http
GET /healthz
```

Returns application health status.

### 2. Model Readiness

```http
GET /readyz
```

Returns model loading status and available labels.

### 3. Text Classification (SVM)

```http
POST /predict-text
Content-Type: application/json

{
    "text": "This Service Level Agreement is made between..."
}
```

### 4. PDF Classification (SVM)

```http
POST /predict-pdf
Content-Type: multipart/form-data

file: [PDF file]
```

### 5. Advanced Classification (LegalBERT)

```http
POST /v1/predict
Content-Type: application/json

{
    "text": "This Service Level Agreement is made between..."
}
```

**Response:**

```json
{
    "prediction": "Service Level Agreement",
    "confidence": 0.95,
    "probabilities": {
        "License Agreement": 0.02,
        "Employment Agreement": 0.01,
        "Service Level Agreement": 0.95,
        ...
    },
    "chunks_used": 3,
    "snippet": "This Service Level Agreement is made between..."
}
```

## 🧠 Model Details

### SVM + TF-IDF Model

- **Purpose**: Fast, lightweight classification
- **Features**: TF-IDF vectorization of cleaned text
- **Algorithm**: Support Vector Machine
- **Use Case**: Quick classification of short to medium-length documents

### LegalBERT Model

- **Purpose**: High-accuracy classification
- **Architecture**: Fine-tuned BERT transformer
- **Features**:
  - Handles long documents through chunking
  - Position embeddings limit: 512 tokens
  - Automatic text chunking with overlap
- **Use Case**: Accurate classification of complex, long legal documents

## 📊 Model Performance

The system includes two trained models:

1. **SVM Classifier**: Fast inference, suitable for real-time applications
2. **LegalBERT Classifier**: Higher accuracy, handles complex legal language

## 🔍 Text Processing Features

### Document Processing

- **PDF Text Extraction**: Supports both text-based and scanned PDFs
- **OCR Integration**: Automatic OCR for scanned documents using Tesseract
- **Text Cleaning**: Removes noise while preserving legal structure
- **Chunking**: Intelligent text chunking for long documents

### Text Cleaning Pipeline

- Lowercase conversion
- URL removal
- HTML tag removal
- Punctuation normalization
- Whitespace normalization
- Date and number preservation

## 🛠️ Development

### Testing the Model

```bash
python test_model.py
```

### Training New Models

1. **SVM Model**: Use `research/trials.ipynb`
2. **BERT Model**: Use `research/legalBert.ipynb`

### Adding New Contract Types

1. Update the model configuration in `src/models/config.json`
2. Retrain the models with new labeled data
3. Update the label mappings in the model files

## 📋 Dependencies

### Core ML Libraries

- `torch==2.0.1` - PyTorch for deep learning
- `transformers==4.31.0` - Hugging Face transformers
- `scikit-learn==1.3.0` - Traditional ML algorithms
- `numpy==1.24.3` - Numerical computing

### Web Framework

- `flask==2.3.2` - Web application framework
- `uvicorn==0.23.2` - ASGI server

### Document Processing

- `pdfplumber==0.9.0` - PDF text extraction
- `pytesseract==0.3.10` - OCR for scanned documents
- `pdf2image==1.16.3` - PDF to image conversion

### Utilities

- `joblib==1.3.2` - Model persistence
- `pandas==2.0.3` - Data manipulation
- `python-multipart==0.0.6` - File upload handling

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**

   - Ensure model files are in `src/models/`
   - Check file permissions
   - Verify model directory path in config

2. **Memory Issues**

   - Reduce `MAX_TOKENS` in config
   - Use smaller batch sizes
   - Enable GPU if available

3. **OCR Issues**

   - Install Tesseract OCR
   - Verify language packs
   - Check image quality

4. **Import Errors**
   - Verify Python path
   - Check relative imports
   - Install missing dependencies

### Debug Mode

Enable debug output by setting environment variables:

```bash
export DEBUG=1
python app.py
```

### Memory Optimization

- Adjust `MAX_TOKENS` based on available memory
- Implement streaming for large files
- Use model quantization for inference
