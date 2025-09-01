import pandas as pd
from src.helper import extract_text_smart,clean_contract_text

def text_processing():
    """Main function to process contract data"""
    try:
        # Try relative path first (when running from utils directory)
        df = pd.read_csv("../data/scontracts.csv")
    except FileNotFoundError:
        # Try absolute path (when running from root directory)
        df = pd.read_csv("data/scontracts.csv")
    
    # print("CSV data:")
    print(df.head())
    
    # # Example of cleaning some text
    # sample_text = "This is a sample contract text with [references] and https://example.com"
    # cleaned_text = clean_contract_text(sample_text)
    # print("\nCleaned text:")
    # print(cleaned_text)
    
    return df

# Run if called directly
if __name__ == "__main__":
    text_processing()



