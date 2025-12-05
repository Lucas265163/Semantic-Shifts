import gensim
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec

# --- Configuration ---
MODELS_DIR = '../models'
OUTPUT_DIR = '../outputs/axis'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERIODS = [
    {
        "name": "Before 2016",
        "rep": os.path.join(MODELS_DIR, "republican_before_2016.model"),
        "dem": os.path.join(MODELS_DIR, "democrats_before_2016.model")
    },
    {
        "name": "2017-2020",
        "rep": os.path.join(MODELS_DIR, "republican_2017_2020.model"),
        "dem": os.path.join(MODELS_DIR, "democrats_2017_2020.model")
    },
    {
        "name": "2021-2024",
        "rep": os.path.join(MODELS_DIR, "republican_2021_2024.model"),
        "dem": os.path.join(MODELS_DIR, "democrats_2021_2024.model")
    }
]

# Ideological Seeds (Right vs Left)
AXIS_SEEDS = [
    ("conservative", "liberal"),
    ("republican", "democrat"),
    ("right", "left")
]

def expand_seeds(model, seeds, topn=5):
    """Expands seed pairs using nearest neighbors."""
    expanded_seeds = list(seeds)
    for r_word, l_word in seeds:
        if r_word in model.wv and l_word in model.wv:
            r_neighbors = [w for w, _ in model.wv.most_similar(r_word, topn=topn)]
            l_neighbors = [w for w, _ in model.wv.most_similar(l_word, topn=topn)]
            for r, l in zip(r_neighbors, l_neighbors):
                expanded_seeds.append((r, l))
    return list(set(expanded_seeds))

def get_processed_vectors(model, words, center=True):
    """Helper to normalize vectors."""
    valid_words = [w for w in words if w in model.wv]
    if not valid_words: return np.array([])
    vecs = np.array([model.wv[word] for word in valid_words])
    if center:
        vecs = vecs - vecs.mean(axis=0)
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1 
    return vecs / norm

def align_models(model_base, model_target):
    """
    Aligns model_target (Dem) to model_base (Rep) using Procrustes Analysis.
    """
    # 1. Identify Common Vocabulary
    common_vocab = list(set(model_base.wv.index_to_key).intersection(model_target.wv.index_to_key))
    
    # Sort by combined frequency
    common_vocab.sort(key=lambda w: model_base.wv.get_vecattr(w, "count") + model_target.wv.get_vecattr(w, "count"), reverse=True)
    
    # 2. Use Top 3000 words as anchors
    anchors = common_vocab[:3000]
    
    # 3. Compute Rotation Matrix
    vecs_base = get_processed_vectors(model_base, anchors, center=True)
    vecs_target = get_processed_vectors(model_target, anchors, center=True)
    
    m = vecs_target.T @ vecs_base
    u, _, vt = np.linalg.svd(m)
    rotation = u @ vt
    
    # 4. Apply to Target Model
    model_target.wv.vectors = model_target.wv.vectors @ rotation
    
    if hasattr(model_target.wv, 'fill_norms'):
        model_target.wv.fill_norms(force=True)
        
    return model_target, common_vocab

def construct_axis_vector(model, seeds):
    """Constructs the vector representing the ideological axis."""
    axis_vectors = []
    for right, left in seeds:
        if right in model.wv and left in model.wv:
            v_r = model.wv[right] / np.linalg.norm(model.wv[right])
            v_l = model.wv[left] / np.linalg.norm(model.wv[left])
            axis_vectors.append(v_r - v_l)
            
    if not axis_vectors:
        return None
    
    final_axis = np.mean(axis_vectors, axis=0)
    return final_axis / np.linalg.norm(final_axis)

def main():
    for period in PERIODS:
        print(f"\n=== Processing Period: {period['name']} ===")
        
        if not os.path.exists(period['rep']) or not os.path.exists(period['dem']):
            print(f"Skipping {period['name']} (models not found)")
            continue

        # 1. Load Models
        model_rep = Word2Vec.load(period['rep'])
        model_dem = Word2Vec.load(period['dem'])
        
        # 2. Align Models (Dem -> Rep)
        print("Aligning models...")
        model_dem, common_vocab = align_models(model_rep, model_dem)
        
        # 3. Construct Axis (Conservative -> Liberal)
        current_seeds = expand_seeds(model_rep, AXIS_SEEDS, topn=5)
        axis_vector = construct_axis_vector(model_rep, current_seeds)
        
        if axis_vector is None:
            print("Error: Could not construct axis from seeds.")
            continue
            
        # 4. Score Vocabulary
        results = []
        # Filter: Only analyze words common to both that are frequent enough (top 80%)
        analysis_vocab = common_vocab[:int(0.8 * len(common_vocab))]
        
        for word in analysis_vocab:
            # Skip short junk words
            if len(word) < 3: continue
                
            v_rep = model_rep.wv[word]
            v_dem = model_dem.wv[word]
            
            # Project onto Axis
            proj_rep = np.dot(v_rep / np.linalg.norm(v_rep), axis_vector)
            proj_dem = np.dot(v_dem / np.linalg.norm(v_dem), axis_vector)
            
            # Polarization = Absolute difference in position on the axis
            polarization = abs(proj_rep - proj_dem)
            
            results.append({
                "word": word,
                "rep_score": proj_rep,
                "dem_score": proj_dem,
                "polarization": polarization
            })
            
        # 5. Save Results
        df = pd.DataFrame(results).sort_values("polarization", ascending=False)
        
        output_file = os.path.join(OUTPUT_DIR, f"{period['name'].replace(' ', '_')}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"Saved analysis to {output_file}")
        print("\nTop 10 Polarized Words:")
        print(df.head(10)[['word', 'polarization', 'rep_score', 'dem_score']])

if __name__ == "__main__":
    main()