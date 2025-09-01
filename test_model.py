#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import LegalBertClassifier
from src.config import Config

def test_model():
    print("Testing model initialization...")
    try:
        classifier = LegalBertClassifier(Config.MODEL_DIR)
        print("Model initialized successfully!")
        
        # Test with a simple text
        test_text = "This Service Level Agreement is made between Company A and Vendor B"
        print(f"\nTesting with text: '{test_text}'")
        
        result = classifier.predict(test_text)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model() 