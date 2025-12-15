import gensim
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


# Configuration
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

# Helper Functions
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def get_neighbors(model, token: str, topn: int) -> List[str]:
    try:
        return [w for w, _ in model.wv.most_similar(token, topn=topn)]
    except KeyError:
        return []


# Ideological Seeds (Right vs Left)
AXIS_SEEDS = [
    ("conservative", "liberal"),
    ("republican", "democrat"),
    ("right", "left")
]

def expand_seeds(
    model,
    human_seeds: List[Tuple[str, str]],
    top_k: int = 50,
    num_pairs: int = 10,
    min_count: int = 50,
    candidate_pool_limit: int = None,
    agg: str = "max",
    score_threshold: float = 0.3,
    verbose: bool = True
) -> List[Tuple[str, str]]:
    
    # Build allowed vocab if requested
    if candidate_pool_limit is not None:
        vocab_sorted = sorted(model.wv.index_to_key, key=lambda w: model.wv.get_vecattr(w, "count"), reverse=True)
        allowed_vocab = set(vocab_sorted[:candidate_pool_limit])
    else:
        allowed_vocab = None

    # Validate & build canonical seed directions
    seed_dirs = []
    valid_human_seeds = []
    for r_seed, l_seed in human_seeds:
        if r_seed in model.wv and l_seed in model.wv:
            v_r = normalize(model.wv[r_seed])
            v_l = normalize(model.wv[l_seed])
            seed_dirs.append(normalize(v_r - v_l))
            valid_human_seeds.append((r_seed, l_seed))
        else:
            if verbose:
                print(f"[expand_seeds_matched] Warning: skipping seed ({r_seed}, {l_seed}) — missing from vocab")

    if len(seed_dirs) == 0:
        raise ValueError("No valid human seeds found in model vocabulary.")

    seed_dirs = np.stack(seed_dirs, axis=0)

    # Retrieve neighbors for each seed pair
    right_neighbors = {}
    left_neighbors = {}
    for r_seed, l_seed in valid_human_seeds:
        r_neigh = get_neighbors(model, r_seed, top_k)
        l_neigh = get_neighbors(model, l_seed, top_k)
        if allowed_vocab is not None:
            r_neigh = [w for w in r_neigh if w in allowed_vocab]
            l_neigh = [w for w in l_neigh if w in allowed_vocab]
        if min_count is not None:
            r_neigh = [w for w in r_neigh if model.wv.get_vecattr(w, "count") >= min_count]
            l_neigh = [w for w in l_neigh if model.wv.get_vecattr(w, "count") >= min_count]
        right_neighbors[(r_seed, l_seed)] = r_neigh
        left_neighbors[(r_seed, l_seed)] = l_neigh

    # Build candidate pairs
    candidate_pairs = set()
    for key in right_neighbors:
        for r in right_neighbors[key]:
            for l in left_neighbors[key]:
                if r == l: continue
                if min_count is not None:
                    if model.wv.get_vecattr(r, "count") < min_count or model.wv.get_vecattr(l, "count") < min_count:
                        continue
                candidate_pairs.add((r, l))
    candidate_pairs = list(candidate_pairs)
    
    if len(candidate_pairs) == 0:
        if verbose: print("No candidate pairs generated.")
        return valid_human_seeds[:num_pairs]

    # Precompute normalized vectors
    vocab_for_cache = set([w for pair in candidate_pairs for w in pair] + [w for sd in valid_human_seeds for w in sd])
    vec_cache = {w: normalize(model.wv[w]) for w in vocab_for_cache if w in model.wv}

    # Score candidates
    records = []
    for (r, l) in candidate_pairs:
        vr = vec_cache.get(r)
        vl = vec_cache.get(l)
        if vr is None or vl is None: continue
        d_cand = normalize(vr - vl)
        sims = seed_dirs.dot(d_cand)
        score = float(np.max(sims)) if agg == "max" else float(np.mean(sims))
        
        records.append({"r": r, "l": l, "score": score,
                        "freq_r": model.wv.get_vecattr(r, "count"),
                        "freq_l": model.wv.get_vecattr(l, "count")})

    df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

    # Greedy selection
    ban_list = [s for sd in valid_human_seeds for s in sd]
    selected = []
    i = 0
    while len(selected) < (num_pairs - len(valid_human_seeds)) and i < len(df):
        row = df.iloc[i]
        if row['r'] in ban_list or row['l'] in ban_list:
            i += 1
            continue
        if row['score'] < score_threshold:
            break
        selected.append((row['r'], row['l'], row['score'], row['freq_r'], row['freq_l']))
        ban_list.extend([row['r'], row['l']])
        i += 1

    final_pairs = list(valid_human_seeds) + [(r, l) for r, l, _, _, _ in selected]
    print(f'final pairs: {final_pairs}')
    return final_pairs[:num_pairs]

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
    # Identify Common Vocabulary
    common_vocab = list(set(model_base.wv.index_to_key).intersection(model_target.wv.index_to_key))
    
    # Sort by combined frequency
    common_vocab.sort(key=lambda w: model_base.wv.get_vecattr(w, "count") + model_target.wv.get_vecattr(w, "count"), reverse=True)
    
    # Use Top 3000 words as anchors
    anchors = common_vocab[:3000]
    
    # Compute Rotation Matrix
    vecs_base = get_processed_vectors(model_base, anchors, center=True)
    vecs_target = get_processed_vectors(model_target, anchors, center=True)
    
    m = vecs_target.T @ vecs_base
    u, _, vt = np.linalg.svd(m)
    rotation = u @ vt
    
    # Apply to Target Model
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

        # Load Models
        model_rep = Word2Vec.load(period['rep'])
        model_dem = Word2Vec.load(period['dem'])
        
        # Align Models (Dem -> Rep)
        print("Aligning models...")
        model_dem, common_vocab = align_models(model_rep, model_dem)
        
        # Construct Axis (Conservative -> Liberal)
        current_seeds = expand_seeds(model_rep, AXIS_SEEDS, topn=5)
        axis_vector = construct_axis_vector(model_rep, current_seeds)
        
        if axis_vector is None:
            print("Error: Could not construct axis from seeds.")
            continue
            
        # Score Vocabulary
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
            
        # Save Results
        df = pd.DataFrame(results).sort_values("polarization", ascending=False)
        
        output_file = os.path.join(OUTPUT_DIR, f"{period['name'].replace(' ', '_')}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"Saved analysis to {output_file}")
        print("\nTop 10 Polarized Words:")
        print(df.head(10)[['word', 'polarization', 'rep_score', 'dem_score']])

if __name__ == "__main__":
    main()