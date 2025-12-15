import os
import sys
import re
import html
import unicodedata
import pickle
import datetime
import spacy
from tqdm import tqdm
import numpy as np

# Attempt to import file_streams depending on how the script is run
try:
    from src.file_streams import getFileJsonStream
except ImportError:
    # Fallback if running directly inside src folder
    from file_streams import getFileJsonStream

# Configuration 
DATA_RAW_DIR = '../data/raw'
DATA_PROCESSED_DIR = '../data/preprocessed'

# Ensure base output directory exists
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

def load_spacy_model():
    """Safely load the spacy model."""
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Error: spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        sys.exit(1)

# Initialize global resources
nlp = load_spacy_model()

def preprocess_text(text, without_stopwords=True):
    """
    Preprocess text content: cleaning, removing artifacts, and lemmatizing.
    """
    # --- Step 1: Text Cleaning (Regex) ---
    if not text or not isinstance(text, str):
        return []

    text = html.unescape(text)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove URLs, Markdown, and references
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'/r/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'/u/\w+', '', text)
    text = re.sub(r'u/\w+', '', text)
    
    # Basic text cleaning (keep only letters and spaces)
    text = re.sub("[^A-Za-z]+", ' ', text).lower()
    
    # --- Step 2: NLP Processing with spaCy ---
    doc = nlp(text)
    
    processed_words = []
    
    for token in doc:
        # Skip whitespace
        if token.is_space:
            continue
            
        # Stopword check
        if without_stopwords and token.is_stop:
            continue
            
        # Get the lemma
        lemma = token.lemma_
        
        # Length check
        if 2 < len(lemma) <= 15:
            processed_words.append(lemma)

    return processed_words


def process_and_save_comments(filename, subreddit, output_dir, without_stopwords=True, batch_size=100000):
    """Process comments and save in batches (chunks)."""
    
    input_path = os.path.join(DATA_RAW_DIR, filename)
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Processing file: {input_path}")
    
    # Batch processing counters
    batch_count = 0
    batch_number = 1
    total_count = 0
    
    comments_batch = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_path, "rb") as f:
            jsonStream = getFileJsonStream(input_path, f)
            if jsonStream is None:
                print(f"Unable to read file stream for {input_path}")
                return
            
            for row in tqdm(jsonStream, desc=f"Processing {subreddit}"):
                # Validate row data
                if not all(key in row for key in ["body", "created_utc", "author", "id"]):
                    continue
                    
                author = row["author"]
                if author in {"AutoModerator", "election_info_bot"}:
                    continue
                
                # Extract Data
                comment_id = row["id"]
                text = row["body"]
                created_timestamp = row["created_utc"]
                try:
                    date = datetime.datetime.fromtimestamp(int(created_timestamp))
                except (ValueError, TypeError):
                    continue
                
                # Process Text
                processed_words = preprocess_text(text, without_stopwords=without_stopwords)
                
                if processed_words:
                    comment_data = {
                        "comment_id": comment_id,
                        "author": author,
                        "date": date.strftime("%Y-%m-%d"),
                        "timestamp": created_timestamp,
                        "processed_text": processed_words,
                        "original": text
                    }
                    
                    comments_batch.append(comment_data)
                    batch_count += 1
                    
                # Save Batch if limit reached
                if batch_count >= batch_size:
                    save_batch(comments_batch, output_dir, subreddit, batch_number)
                    
                    # Reset batch data
                    comments_batch = []
                    batch_count = 0
                    batch_number += 1
                    total_count += batch_size
        
        # Process any remaining comments
        if batch_count > 0:
            save_batch(comments_batch, output_dir, subreddit, batch_number)
            total_count += batch_count
        
        print(f"\nCompleted {subreddit}! Total saved: {total_count}")

    except Exception as e:
        print(f"An error occurred processing {filename}: {e}")

def save_batch(data, output_dir, subreddit, batch_number):
    """Helper function to save a batch of data."""
    save_path = os.path.join(output_dir, f"{subreddit}_batch{batch_number}.pkl")
    print(f"\nSaving batch {batch_number} to {save_path}...")
    with open(save_path, "wb") as out_file:
        pickle.dump(data, out_file)
    print(f"Saved {len(data)} comments.")

def main():
    """Main execution function."""
    # Map your specific filenames here
    # These filenames must exist in ../data/raw
    files = {
        "democrats": "democrats_comments.zst",
        "republican": "Republican_comments.zst"
    }
    
    # Ensure base output directory exists
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    for subreddit, filename in files.items():
        subreddit_output_dir = os.path.join(DATA_PROCESSED_DIR, subreddit)
        
        print(f"\n--- Starting {subreddit} ---")
        process_and_save_comments(
            filename=filename,
            subreddit=subreddit,
            output_dir=subreddit_output_dir,
            without_stopwords=True,
            batch_size=100000 # Adjusted to a safer default for memory
        )

if __name__ == "__main__":
    main()