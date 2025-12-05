import pickle
import datetime
import re
import html
import unicodedata
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import random
import sys
import os
sys.path.append(os.path.abspath('..\\..'))  # Adjust path if needed based on your execution context
from file_streams import getFileJsonStream
import spacy
import re
import html
import unicodedata


# Initialize global resources once
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# POS tag cache to avoid redundant tagging
POS_CACHE = {}
# Lemma cache to avoid redundant lemmatization
LEMMA_CACHE = {}

def get_wordnet_pos(tag):
    """Convert NLTK POS tag to WordNet POS tag"""
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default as noun


def preprocess_text_spacy(text, without_stopwords=True):
    """
    Preprocess Reddit text content using spaCy.
    """
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("Error: spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise
    # --- Step 1: Text Cleaning (Regex) ---
    # Keep your original cleaning logic; it's faster to do this with regex 
    # before passing it to spaCy.
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
    
    # Basic text cleaning
    text = re.sub("[^A-Za-z]+", ' ', text).lower()
    
    # --- Step 2: NLP Processing with spaCy ---
    
    # nlp(text) tokenizes, tags, and lemmatizes the text
    doc = nlp(text)
    
    processed_words = []
    
    for token in doc:
        # Skip whitespace (though regex mostly handled this)
        if token.is_space:
            continue
            
        # Stopword check using spaCy's built-in list
        if without_stopwords and token.is_stop:
            continue
            
        # Get the lemma
        lemma = token.lemma_
        
        # Length check (from your original requirements)
        if 2 < len(lemma) <= 15:
            processed_words.append(lemma)

    return processed_words


def process_and_save_comments(path, subreddit, output_dir, without_stopwords=True, batch_size=1000000):
    """Process comments and save in batches"""
    print(f"Processing file: {path}")
    
    # Batch processing counters
    batch_count = 0
    batch_number = 1
    total_count = 0
    
    # Create data structure for comments
    comments_batch = []

    with open(path, "rb") as f:
        jsonStream = getFileJsonStream(path, f)
        if jsonStream is None:
            print(f"Unable to read file {path}")
            return
        
        for row in tqdm(jsonStream, desc=f"Processing {subreddit} comments"):
            if "body" not in row or "created_utc" not in row or "author" not in row or "id" not in row:
                continue
                
            author = row["author"]
            if author in {"AutoModerator", "election_info_bot"}:
                continue
            
            comment_id = row["id"]
            text = row["body"]
            created_timestamp = row["created_utc"]
            date = datetime.datetime.fromtimestamp(int(created_timestamp))
            
            # Process text with optimized functions
            processed_words = preprocess_text(text, lemmatize=True, without_stopwords=without_stopwords)
            
            if processed_words:
                # Save processed comment with metadata
                comment_data = {
                    "comment_id": comment_id,
                    "author": author,
                    "date": date.strftime("%Y-%m-%d"),
                    "timestamp": created_timestamp,
                    "processed_text": processed_words,  # Original order preserved
                    "original": text
                }
                
                comments_batch.append(comment_data)
                batch_count += 1
                
            # Check if we need to save the current batch
            if batch_count >= batch_size:
                print(f"\nReached {batch_size} comments, saving batch {batch_number}...")
                
                # Save batch directly without filtering
                save_path = f"{output_dir}/{subreddit}_batch{batch_number}.pkl"
                with open(save_path, "wb") as out_file:
                    pickle.dump(comments_batch, out_file)
                
                print(f"Saved {len(comments_batch)} comments to {save_path}")
                
                # Reset batch data
                comments_batch = []
                batch_count = 0
                batch_number += 1
                total_count += batch_size
    
    # Process any remaining comments
    if batch_count > 0:
        print(f"\nSaving remaining {batch_count} comments...")
        
        # Save batch
        save_path = f"{output_dir}/{subreddit}_batch{batch_number}.pkl"
        with open(save_path, "wb") as out_file:
            pickle.dump(comments_batch, out_file)
        
        print(f"Saved {len(comments_batch)} comments to {save_path}")
        total_count += batch_count
    
    print(f"\nCompleted processing {subreddit} comments!")
    print(f"Total comments saved: {total_count}")


def main():
    """Main function"""
    random.seed(23)
    np.random.seed(23)
    
    # Define data file paths
    files = {
        "democrats": r"datasets/democrats_comments.zst",
        "republican": r"datasets/Republican_comments.zst",
        "conservative": r"datasets/Conservative_comments.zst",
        "liberal": r"datasets/Liberal_comments.zst"
    }
    
    # List of subreddits to process (process all by default)
    subreddits_to_process = list(files.keys())

    for subreddit in subreddits_to_process:
        output_dir = f"processed_comments_2/{subreddit}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing subreddit: {subreddit}")
        process_and_save_comments(
            files[subreddit],
            subreddit,
            output_dir,
            without_stopwords=False,
            batch_size=1000000
        )