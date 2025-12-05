import os
import glob
import pickle
import datetime
import random
import numpy as np
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

# --- Configuration ---
DATA_PROCESSED_DIR = '/data/preprocessed'
MODELS_DIR = '../models'
INTERIM_MODELS_DIR = os.path.join(MODELS_DIR, 'interim')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(INTERIM_MODELS_DIR, exist_ok=True)

def get_date_from_comment(comment):
    """Extract date object from a comment dictionary."""
    try:
        # Try string format first
        return datetime.datetime.strptime(comment["date"], "%Y-%m-%d").date()
    except (KeyError, ValueError, TypeError):
        try:
            # Fallback to timestamp
            return datetime.datetime.fromtimestamp(int(comment["timestamp"])).date()
        except (KeyError, ValueError, TypeError):
            return None

def get_period(date):
    """Determine which time period a date belongs to."""
    if date is None:
        return None
    year = date.year
    if year <= 2016:
        return "before_2016"
    elif 2017 <= year <= 2020:
        return "2017_2020"
    elif 2021 <= year <= 2024:
        return "2021_2024"
    return None

def train_global_bigram_model(subreddits, output_path, min_count=10, threshold=10):
    """
    Trains a Phraser model on all available data to detect common bigrams 
    (e.g., 'new_york', 'machine_learning').
    """
    if os.path.exists(output_path):
        print(f"Global bigram model already exists at {output_path}. Skipping training.")
        return Phraser.load(output_path)

    print("\n--- Starting Global Bigram Training ---")
    phrases = Phrases(min_count=min_count, threshold=threshold)
    total_sentences = 0

    for subreddit in subreddits:
        # Match the batch files created in preprocessing
        pattern = os.path.join(DATA_PROCESSED_DIR, subreddit, f"{subreddit}_batch*.pkl")
        files = sorted(glob.glob(pattern))
        
        print(f"Loading {len(files)} files for subreddit: {subreddit}")
        
        for file_path in files:
            try:
                with open(file_path, "rb") as f:
                    comments = pickle.load(f)
                
                batch_sentences = [
                    c["processed_text"] for c in comments 
                    if "processed_text" in c and c["processed_text"]
                ]
                
                if batch_sentences:
                    phrases.add_vocab(batch_sentences)
                    total_sentences += len(batch_sentences)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Total sentences for bigram training: {total_sentences}")
    
    if total_sentences == 0:
        print("No data found! Check your preprocessed data paths.")
        return None

    bigram_model = Phraser(phrases)
    bigram_model.save(output_path)
    print(f"Global bigram model saved to {output_path}")
    return bigram_model

def apply_bigrams(comments_list, bigram_model):
    """Apply the trained bigram model to a list of sentences."""
    processed = []
    for comment in comments_list:
        if "processed_text" in comment and comment["processed_text"]:
            processed.append(bigram_model[comment["processed_text"]])
    return processed

def create_or_update_w2v_model(period, sentences, vector_size, window, min_count, workers, sg, epochs, existing_model=None):
    """Creates a new Word2Vec model or updates an existing one with new data."""
    if not sentences:
        return existing_model

    if existing_model is None:
        print(f"[{period}] Initializing new Word2Vec model...")
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            seed=23
        )
        model.build_vocab(sentences)
    else:
        print(f"[{period}] Updating existing model...")
        model = existing_model
        model.build_vocab(sentences, update=True)
    
    print(f"[{period}] Training on {len(sentences)} sentences...")
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model

def build_models_for_subreddit(subreddit, global_bigram_model, config):
    """
    Trains Word2Vec models for a specific subreddit, split by time periods.
    """
    print(f"\n--- Building Models for: {subreddit} ---")
    
    time_periods = ["before_2016", "2017_2020", "2021_2024"]
    models = {period: None for period in time_periods}
    
    # Buffers to hold data until we reach chunk_size
    comments_buffer = {period: [] for period in time_periods}
    
    # Find input files
    pattern = os.path.join(DATA_PROCESSED_DIR, subreddit, f"{subreddit}_batch*.pkl")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No data files found for {subreddit}")
        return

    for file_path in files:
        print(f"Processing file: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'rb') as f:
                comments = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        # Sort comments into time periods
        for comment in comments:
            date = get_date_from_comment(comment)
            period = get_period(date)
            if period:
                comments_buffer[period].append(comment)

        # Check if any buffer is full enough to train
        for period in time_periods:
            if len(comments_buffer[period]) >= config['chunk_size']:
                chunk = comments_buffer[period]
                # Clear buffer
                comments_buffer[period] = []
                
                # Apply Bigrams
                processed_chunk = apply_bigrams(chunk, global_bigram_model)
                
                # Train/Update Model
                models[period] = create_or_update_w2v_model(
                    period, processed_chunk, 
                    config['vector_size'], config['window'], config['min_count'], 
                    config['workers'], config['sg'], config['epochs'], 
                    models[period]
                )
                
                # Save interim model
                save_path = os.path.join(INTERIM_MODELS_DIR, f"{subreddit}_{period}_interim.model")
                models[period].save(save_path)

    # Process remaining data in buffers
    print("Processing remaining data...")
    for period, remaining_comments in comments_buffer.items():
        if len(remaining_comments) > config['min_comments_to_train']:
            processed_chunk = apply_bigrams(remaining_comments, global_bigram_model)
            models[period] = create_or_update_w2v_model(
                period, processed_chunk, 
                config['vector_size'], config['window'], config['min_count'], 
                config['workers'], config['sg'], config['epochs'], 
                models[period]
            )
        else:
            print(f"[{period}] Skipping final chunk (only {len(remaining_comments)} comments)")

    # Save Final Models
    for period, model in models.items():
        if model is not None:
            final_path = os.path.join(MODELS_DIR, f"{subreddit}_{period}.model")
            model.save(final_path)
            print(f"Saved final model: {final_path}")

def main():
    # Set seeds for reproducibility
    random.seed(23)
    np.random.seed(23)
    
    # Configuration
    # You can modify the list of subreddits here
    subreddits = ["democrats", "republican"]
    
    training_config = {
        "vector_size": 300,
        "window": 5,
        "min_count": 10,
        "epochs": 5,
        "workers": 16,  # Adjust based on your CPU cores
        "sg": 0,        # 0 = CBOW, 1 = Skip-gram
        "min_comments_to_train": 5000,
        "chunk_size": 50000  # Process 50k comments at a time to save RAM
    }

    # 1. Train Global Bigram Model
    bigram_model_path = os.path.join(MODELS_DIR, "global_bigram.phr")
    global_bigram_model = train_global_bigram_model(
        subreddits, 
        bigram_model_path
    )

    if not global_bigram_model:
        print("Failed to initialize bigram model. Exiting.")
        return

    # 2. Train Word2Vec Models per Subreddit
    for subreddit in subreddits:
        build_models_for_subreddit(subreddit, global_bigram_model, training_config)

if __name__ == "__main__":
    main()