import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# --- Files ---
# Map subreddit names to your raw filenames
DATA_FILES = {
    "democrats": "democrats_comments.zst",
    "republican": "Republican_comments.zst",
    # Add others here
}

# --- Model Parameters ---
VECTOR_SIZE = 300
WINDOW = 5
MIN_COUNT = 10
WORKERS = 4
EPOCHS = 5

# --- Time Periods ---
def get_period(year):
    if year <= 2016:
        return "Before_2016"
    elif 2017 <= year <= 2020:
        return "2017_2020"
    elif 2021 <= year <= 2024:
        return "2021_2024"
    return None