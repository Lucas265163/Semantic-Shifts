import os
import pickle
import datetime
import glob
import numpy as np
import random
from gensim.models.phrases import Phraser
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

import glob
import pickle
from gensim.models.phrases import Phrases, Phraser
import os

def train_and_save_global_bigram_model(subreddits, base_data_dir, output_path, min_count=10, threshold=10):
    phrases = Phrases(min_count=min_count, threshold=threshold)
    total_sentences = 0

    for subreddit in subreddits:
        pattern = f"{base_data_dir}/{subreddit}/{subreddit}_batch*.pkl"
        files = sorted(glob.glob(pattern))
        print(f"Pattern: {files}")
        print(f"Loading {len(files)} files for subreddit: {subreddit}")
        for file_path in files:
            try:
                with open(file_path, "rb") as f:
                    comments = pickle.load(f)
                batch_sentences = [
                    comment["processed_text"]
                    for comment in comments
                    if "processed_text" in comment
                ]
                phrases.add_vocab(batch_sentences)
                total_sentences += len(batch_sentences)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Total sentences for bigram training: {total_sentences}")
    bigram_model = Phraser(phrases)
    bigram_model.save(output_path)
    print(f"Global bigram model saved to {output_path}")

def main():
    subreddits = ["democrats", "republican", "conservative", "liberal", "technology", "cooking", "movies", "books", "personalfinance", "travel"]

    output_dir = "models/bigram"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/all_bigram_1.phr"
    base_data_dir = "processed_comments_1"

    train_and_save_global_bigram_model(
        subreddits,
        base_data_dir=base_data_dir,
        output_path=output_path,
        min_count=10,
        threshold=10
    )

if __name__ == "__main__":
    main()

def get_date_from_comment(comment):
    """Extract date from a comment dictionary"""
    try:
        return datetime.datetime.strptime(comment["date"], "%Y-%m-%d").date()
    except (KeyError, ValueError):
        try:
            return datetime.datetime.fromtimestamp(int(comment["timestamp"])).date()
        except (KeyError, ValueError):
            return None

def get_period(date):
    """Determine which time period a date belongs to"""
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

def build_bigram_model(comments):
    """Build a bigram model for the given comments"""
    sentences = []
    for comment in comments:
        if "processed_text" in comment:
            sentences.append(comment["processed_text"])
    phrases = Phrases(sentences, min_count=10, threshold=10)
    return Phraser(phrases)

def apply_bigrams(comments, bigram_model):
    """Apply bigram model to comments"""
    processed = []
    for comment in comments:
        if "processed_text" in comment:
            processed.append(bigram_model[comment["processed_text"]])
    return processed

def create_or_update_model(period, comments, vector_size, window, min_count, workers, sg, epochs, existing_model=None):
    """Create a new model or update an existing one"""
    if existing_model is None:
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            seed=23
        )
        model.build_vocab(comments)
        print(f"{period} vocabulary size: {len(model.wv.index_to_key)}")
    else:
        model = existing_model
        model.build_vocab(comments, update=True)
        print(f"{period} vocabulary size: {len(model.wv.index_to_key)}")
    model.train(comments, total_examples=len(comments), epochs=epochs)
    return model

def save_model(model, subreddit, period, model_dir, is_interim=False):
    """Save model to disk"""
    if is_interim:
        path = f"{model_dir}/interim/{subreddit}_{period}_interim.model"
    else:
        path = f"{model_dir}/{subreddit}_{period}.model"
    model.save(path)

def build_models_for_subreddit(
    subreddit,
    base_data_dir,
    model_dir,
    vector_size=300,
    window=5,
    min_count=5,
    epochs=5,
    workers=16,
    sg=0,
    min_comments_to_train=10000,
    chunk_size=1000000,
    global_bigram_path=None
):

    time_periods = ["before_2016", "2017_2020", "2021_2024"]
    models = {period: None for period in time_periods}
    bigram_models = {period: None for period in time_periods}
    
    # Load global bigram model if exists
    global_bigram_path = global_bigram_path
    if os.path.exists(global_bigram_path):
        print(f"Loading global bigram model from {global_bigram_path}")
        global_bigram_model = Phraser.load(global_bigram_path)
    else:
        print(f"Global bigram model not found at {global_bigram_path}, will train on each chunk.")
        global_bigram_model = None
        return

    # Find all pickle files
    pattern = f"{base_data_dir}/{subreddit}/{subreddit}_batch*.pkl"
    pickle_files = sorted(glob.glob(pattern))
    if not pickle_files:
        print(f"No pickle files found for {subreddit} in {base_data_dir}/{subreddit}/")
        return

    comments_by_period = {period: [] for period in time_periods}

    for file_path in pickle_files:
        try:
            with open(file_path, 'rb') as f:
                comments = pickle.load(f)
            print(f"Loaded {len(comments)} comments from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        for comment in comments:
            date = get_date_from_comment(comment)
            period = get_period(date)
            if period:
                comments_by_period[period].append(comment)

        for period in time_periods:
            period_comments = comments_by_period[period]
            while len(period_comments) >= chunk_size:
                print(f"Processing chunk of {chunk_size} comments for {period}")
                chunk = period_comments[:chunk_size]
                period_comments = period_comments[chunk_size:]

                # Use global bigram model if exists, otherwise train on each chunk
                if global_bigram_model is not None:
                    bigram_model = global_bigram_model
                else:
                    bigram_model = build_bigram_model(chunk)
                bigram_models[period] = bigram_model
                processed_chunk = apply_bigrams(chunk, bigram_model)

                if len(processed_chunk) > min_comments_to_train:
                    model = create_or_update_model(
                        period, processed_chunk, vector_size, window, min_count, workers, sg, epochs, models[period]
                    )
                    models[period] = model
                    save_model(model, subreddit, period, model_dir, is_interim=True)
            comments_by_period[period] = period_comments

    # Process any remaining comments
    for period, remaining_comments in comments_by_period.items():
        if len(remaining_comments) > min_comments_to_train:
            print(f"Processing final {len(remaining_comments)} comments for {period}")
            if global_bigram_model is not None:
                bigram_model = global_bigram_model
            else:
                bigram_model = build_bigram_model(remaining_comments)
            bigram_models[period] = bigram_model
            processed_chunk = apply_bigrams(remaining_comments, bigram_model)
            model = create_or_update_model(
                period, processed_chunk, vector_size, window, min_count, workers, sg, epochs, models[period]
            )
            models[period] = model
            save_model(model, subreddit, period, model_dir, is_interim=False)
        else:
            print(f"Skipping final {len(remaining_comments)} comments for {period} (less than minimum required)")

    # Save final models
    for period, model in models.items():
        if model is not None:
            save_model(model, subreddit, period, model_dir, is_interim=False)
    print(f"Model saved to {model_dir}")
    print(f"Completed building models for {subreddit}")

def main():
    model_dir = "models"
    global_bigram_path = "models/bigram.phr"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}/interim", exist_ok=True)
    random.seed(23)
    np.random.seed(23)
    subreddits = ["democrats", "republican"]
    for subreddit in subreddits:
        build_models_for_subreddit(
            subreddit,
            base_data_dir="processed_comments_1",
            model_dir=model_dir,
            vector_size=300,
            window=5,
            min_count=10,
            epochs=5,
            workers=16,
            sg=0,
            min_comments_to_train=10000,
            chunk_size=1000000,
            global_bigram_path=global_bigram_path
        )

if __name__ == "__main__":
    main()