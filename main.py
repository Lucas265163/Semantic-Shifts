import os
import argparse
from src import preprocessing, training, alignment, visualization, analysis, config
from gensim.models import Word2Vec

def main():
    parser = argparse.ArgumentParser(description="Reddit Semantic Drift Analysis Pipeline")
    parser.add_argument('--step', type=str, required=True, 
                        choices=['preprocess', 'train', 'align', 'visualize'],
                        help="Which step of the pipeline to run")
    parser.add_argument('--subreddit', type=str, default='democrats',
                        help="Subreddit to process")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if args.step == 'preprocess':
        print("--- Step 1: NLP Pipeline (spaCy) ---")
        preprocessing.process_subreddit(args.subreddit)
        
    elif args.step == 'train':
        print("--- Step 2: Temporal Model Training ---")
        phraser = training.train_phraser(args.subreddit)
        training.train_temporal_models(args.subreddit, phraser)
        
    elif args.step == 'align':
        print("--- Step 3: Orthogonal Procrustes Alignment ---")
        # Example: Align 2016 model to 2024 model
        path_old = os.path.join(config.MODELS_DIR, f"{args.subreddit}_Before_2016.model")
        path_new = os.path.join(config.MODELS_DIR, f"{args.subreddit}_2021_2024.model")
        
        m_old = Word2Vec.load(path_old)
        m_new = Word2Vec.load(path_new)
        
        m_base, m_aligned, vocab = alignment.intersection_align_gensim(m_new, m_old)
        
        # Save aligned model
        m_aligned.save(os.path.join(config.MODELS_DIR, f"{args.subreddit}_Before_2016_ALIGNED.model"))
        print("Alignment complete.")

    elif args.step == 'visualize':
        print("--- Step 4: Visualization (UMAP + HDBSCAN) ---")
        model_path = os.path.join(config.MODELS_DIR, f"{args.subreddit}_2021_2024.model")
        model = Word2Vec.load(model_path)
        
        # Visualize top 2000 words
        words = model.wv.index_to_key[:2000]
        visualization.plot_embedding_cluster(model, words)

if __name__ == "__main__":
    main()