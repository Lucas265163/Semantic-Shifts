import os
import argparse
from src import preprocessing, training, distance_analysis, axis_analysis, visualization

def main():
    parser = argparse.ArgumentParser(description="Reddit Semantic Shift Analysis Pipeline")
    
    # Main command selector
    parser.add_argument('step', type=str, 
                        choices=['preprocess', 'train', 'distance', 'axis', 'visualize'],
                        help="The pipeline step to execute.")
    
    # Optional arguments
    parser.add_argument('--subreddit', type=str, 
                        help="Subreddit name (required for preprocess, optional for train)")
    parser.add_argument('--file', type=str, 
                        help="Input filename located in data/raw (required for preprocess)")
    parser.add_argument('--model', type=str, 
                        help="Model filename located in models/ (required for visualize)")
    
    args = parser.parse_args()
    
    # Preprocessing
    if args.step == 'preprocess':
        if not args.subreddit or not args.file:
            print("Error: 'preprocess' step requires --subreddit and --file arguments.")
            print("Example: python main.py preprocess --subreddit democrats --file democrats_comments.zst")
            return
        
        # Construct output path dynamically
        output_dir = os.path.join(preprocessing.DATA_PROCESSED_DIR, args.subreddit)
        
        print(f"--- Starting Preprocessing for r/{args.subreddit} ---")
        print(f"Input File: {args.file}")
        
        preprocessing.process_and_save_comments(
            filename=args.file,
            subreddit=args.subreddit,
            output_dir=output_dir,
            without_stopwords=True
        )

    # Training 
    elif args.step == 'train':
        print("--- Starting Model Training ---")
        
        # Training Configuration
        config = {
            "vector_size": 300,
            "window": 5,
            "min_count": 10,
            "epochs": 5,
            "workers": 16,  # Adjust based on your CPU
            "sg": 0,        # CBOW
            "min_comments_to_train": 5000,
            "chunk_size": 50000
        }

        # Determine scope
        if args.subreddit:
            subreddits = [args.subreddit]
        else:
            # Default list if none specified
            subreddits = ["democrats", "republican", "conservative", "liberal"]
        
        # Global Bigram Model
        bigram_path = os.path.join(training.MODELS_DIR, "global_bigram.phr")
        global_bigram_model = training.train_global_bigram_model(subreddits, bigram_path)

        if not global_bigram_model:
            print("Error: Could not train or load global bigram model.")
            return

        # Train Word2Vec for each subreddit
        for sub in subreddits:
            training.build_models_for_subreddit(sub, global_bigram_model, config)

    # Distance Analysis (Polarization)
    elif args.step == 'distance':
        print("--- Starting Distance Analysis (Republican vs Democrat) ---")
        distance_analysis.main()

    # Axis Analysis (Ideological Placement)
    elif args.step == 'axis':
        print("--- Starting Axis Analysis (Conservative <-> Liberal) ---")
        axis_analysis.main()

    # Visualization
    elif args.step == 'visualize':
        if not args.model:
            print("Error: 'visualize' step requires --model argument.")
            print("Example: python main.py visualize --model democrats_2021_2024.model")
            return
            
        print(f"--- Visualizing Model: {args.model} ---")
        visualization.visualize_embedding(args.model)

if __name__ == "__main__":
    main()