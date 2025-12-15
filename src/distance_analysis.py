import os
import gensim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gensim.models import Word2Vec

# Configuration 
MODELS_DIR = '../models'
OUTPUT_DIR = '../plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define which models to compare
PERIODS = [
    {
        "name": "Before 2016",
        "rep": os.path.join(MODELS_DIR, "republican_before_2016.model"),
        "dem": os.path.join(MODELS_DIR, "democrats_before_2016.model"),
        "color": "green"
    },
    {
        "name": "2017-2020",
        "rep": os.path.join(MODELS_DIR, "republican_2017_2020.model"),
        "dem": os.path.join(MODELS_DIR, "democrats_2017_2020.model"),
        "color": "orange"
    },
    {
        "name": "2021-2024",
        "rep": os.path.join(MODELS_DIR, "republican_2021_2024.model"),
        "dem": os.path.join(MODELS_DIR, "democrats_2021_2024.model"),
        "color": "purple"
    }
]

def get_processed_vectors(model, words, center=True):
    """Get normalized vectors for a list of words."""
    valid_words = [w for w in words if w in model.wv]
    if not valid_words:
        return np.array([])
        
    vecs = np.array([model.wv[word] for word in valid_words])
    
    if center:
        vecs = vecs - vecs.mean(axis=0)
        
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1 
    return vecs / norm

def align_and_measure(path_rep, path_dem):
    # Load
    print(f"Processing {path_rep}...")
    model_rep = gensim.models.Word2Vec.load(path_rep)
    model_dem = gensim.models.Word2Vec.load(path_dem)

    # Rough Alignment
    # Filter to top 60% by frequency in each model
    rep_vocab_sorted = sorted(model_rep.wv.index_to_key, key=lambda w: model_rep.wv.get_vecattr(w, "count"), reverse=True)
    dem_vocab_sorted = sorted(model_dem.wv.index_to_key, key=lambda w: model_dem.wv.get_vecattr(w, "count"), reverse=True)
    
    num_rep = int(0.6 * len(rep_vocab_sorted))
    num_dem = int(0.6 * len(dem_vocab_sorted))
    
    top_rep = set(rep_vocab_sorted[:num_rep])
    top_dem = set(dem_vocab_sorted[:num_dem])
    
    common_vocab = list(top_rep.intersection(top_dem))
    
    # Sort by frequency sum
    common_vocab.sort(key=lambda w: model_rep.wv.get_vecattr(w, "count") + model_dem.wv.get_vecattr(w, "count"), reverse=True)
    
    # Anchors
    initial_anchors = common_vocab[:3000]
    
    # First Rotation
    vecs_rep_rough = get_processed_vectors(model_rep, initial_anchors, center=True)
    vecs_dem_rough = get_processed_vectors(model_dem, initial_anchors, center=True)
    
    m = vecs_dem_rough.T @ vecs_rep_rough
    u, _, vt = np.linalg.svd(m)
    rotation_1 = u @ vt
    vecs_dem_rotated = vecs_dem_rough @ rotation_1
    
    # Filter Anchors
    similarities = np.sum(vecs_rep_rough * vecs_dem_rotated, axis=1)
    distances = 1 - similarities
    anchor_scores = sorted(zip(initial_anchors, distances), key=lambda x: x[1])
    refined_anchors = [w for w, d in anchor_scores[:1500]]
    
    # Final Alignment
    vecs_rep_final = get_processed_vectors(model_rep, refined_anchors, center=True)
    vecs_dem_final = get_processed_vectors(model_dem, refined_anchors, center=True)
    
    m_final = vecs_dem_final.T @ vecs_rep_final
    u_final, _, vt_final = np.linalg.svd(m_final)
    rotation_final = u_final @ vt_final
    
    # Apply Rotation & Translation
    model_dem.wv.vectors = model_dem.wv.vectors @ rotation_final
    
    mean_rep = np.mean(model_rep.wv[refined_anchors], axis=0)
    mean_dem = np.mean(model_dem.wv[refined_anchors], axis=0)
    model_dem.wv.vectors = model_dem.wv.vectors + (mean_rep - mean_dem)
    
    if hasattr(model_dem.wv, 'fill_norms'):
        model_dem.wv.fill_norms(force=True)
    
    # Measure Distances
    # Take top 80% by frequency to avoid noise in plotting
    num_anchors = int(0.8 * len(common_vocab))
    analysis_vocab = common_vocab[:num_anchors]
    
    vecs_rep_all = get_processed_vectors(model_rep, analysis_vocab, center=True)
    vecs_dem_all = get_processed_vectors(model_dem, analysis_vocab, center=True)
    
    all_distances = 1 - np.sum(vecs_rep_all * vecs_dem_all, axis=1)
    
    return all_distances, analysis_vocab, (model_rep, model_dem)

def plot_results(results):
    """Generates and saves the polarization plot."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Semantic Polarization Over Time (Republican vs Democrat)', fontsize=20)
    
    axes_flat = axes.flatten()
    
    # Individual Period Plots
    for i, period in enumerate(PERIODS):
        if period["name"] not in results:
            continue
            
        dists = results[period["name"]]
        name = period["name"]
        color = period["color"]
        
        ax = axes_flat[i]
        sns.kdeplot(dists, ax=ax, fill=True, color=color, label=name, linewidth=2)
        
        mean_val = np.mean(dists)
        ax.axvline(mean_val, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
        
        ax.set_title(f'Period: {name}', fontsize=14)
        ax.set_xlabel('Cosine Distance')
        ax.set_xlim(0, 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Overlay Plot
    ax_final = axes_flat[3]
    for period in PERIODS:
        if period["name"] in results:
            dists = results[period["name"]]
            sns.kdeplot(dists, ax=ax_final, fill=False, linewidth=3, 
                       label=f'{period["name"]} (μ={np.mean(dists):.2f})', 
                       color=period["color"])
    
    ax_final.set_title('Overlay Comparison', fontsize=14, fontweight='bold')
    ax_final.set_xlabel('Cosine Distance (Shift to Right = More Polarization)')
    ax_final.set_xlim(0, 1.2)
    ax_final.legend()
    ax_final.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "polarization_plot.png")
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

def analyze_top_drivers(vocab, distances, model_rep, model_dem, period_name):
    """Prints the words that are driving the polarization."""
    print(f"\n--- Top Polarization Drivers ({period_name}) ---")
    
    df = pd.DataFrame({'word': vocab, 'distance': distances})
    # Sort by highest distance
    top_drivers = df.sort_values('distance', ascending=False).head(20)
    
    for index, row in top_drivers.iterrows():
        word = row['word']
        dist = row['distance']
        
        try:
            # Get nearest neighbors from both models
            neighbors_rep = [w for w, s in model_rep.wv.most_similar(word, topn=5)]
            neighbors_dem = [w for w, s in model_dem.wv.most_similar(word, topn=5)]
            
            print(f"\nWord: {word.upper()} (Distance: {dist:.3f})")
            print(f"  Rep context: {', '.join(neighbors_rep)}")
            print(f"  Dem context: {', '.join(neighbors_dem)}")
        except KeyError:
            continue

def main():
    results = {}
    
    # Store the last period's models for detailed analysis
    last_period_data = None
    
    for period in PERIODS:
        print(f"\n--- Analyzing Period: {period['name']} ---")
        dists, vocab, models = align_and_measure(period["rep"], period["dem"])
        
        if dists is not None:
            results[period["name"]] = dists
            last_period_data = (vocab, dists, models, period["name"])
        else:
            print(f"Skipping {period['name']} due to missing data.")

    if results:
        plot_results(results)
    
    # Analyze the drivers for the most recent period available
    if last_period_data:
        vocab, dists, (model_rep, model_dem), name = last_period_data
        analyze_top_drivers(vocab, dists, model_rep, model_dem, name)

if __name__ == "__main__":
    main()