import os
import gensim
import numpy as np
import pandas as pd
import umap
import hdbscan
import plotly.express as px
import plotly.io as pio
from gensim.models import Word2Vec

# Configuration 
MODELS_DIR = '../models'
OUTPUT_DIR = '../plots'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_name):
    """Safe model loader."""
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return None
    return Word2Vec.load(path)

def get_word_data(model, min_count=50):
    """
    Extracts vectors, words, and frequencies from the model.
    Filters out rare words to keep the visualization clean.
    """
    words = []
    vectors = []
    frequencies = []
    
    for word in model.wv.index_to_key:
        count = model.wv.get_vecattr(word, "count")
        if count < min_count:
            continue
            
        words.append(word)
        vectors.append(model.wv[word])
        frequencies.append(count)
        
    return np.array(words), np.array(vectors), np.array(frequencies)

def reduce_dimensions_umap(vectors, n_neighbors=15, min_dist=0.1):
    """
    Reduces vector dimensions to 2D using UMAP.
    """
    print(f"Running UMAP on {len(vectors)} vectors...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    embedding = reducer.fit_transform(vectors)
    return embedding

def cluster_vectors_hdbscan(vectors, min_cluster_size=15):
    """
    Clusters vectors using HDBSCAN (Hierarchical Density-Based Spatial Clustering).
    Does not require specifying the number of clusters k.
    """
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean', 
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(vectors)
    return labels

def analyze_clusters(df, n_top_words=10):
    """
    Prints the top words for each cluster to help identify themes.
    """
    print("\n--- Cluster Analysis ---")
    unique_clusters = sorted(df['cluster'].unique())
    
    cluster_summaries = []
    
    for label in unique_clusters:
        if label == -1:
            continue # Skip noise
            
        cluster_words = df[df['cluster'] == label]
        # Sort by frequency to find representative words
        top_words = cluster_words.sort_values('frequency', ascending=False).head(n_top_words)
        
        words_str = ", ".join(top_words['word'].tolist())
        print(f"Cluster {label} ({len(cluster_words)} words): {words_str}")
        
        cluster_summaries.append({
            "cluster": label,
            "top_words": words_str,
            "size": len(cluster_words)
        })
        
    return pd.DataFrame(cluster_summaries)

def visualize_embedding(model_name, min_count=100):
    """
    Main pipeline: Load -> UMAP -> HDBSCAN -> Interactive Plot
    """
    print(f"\nVisualizing {model_name}...")
    model = load_model(model_name)
    if not model:
        return

    # Prepare Data
    words, vectors, freqs = get_word_data(model, min_count=min_count)
    
    # UMAP Reduction (High dim -> 2D)
    # Note: We cluster on the UMAP projection for cleaner visual clusters,
    # though clustering on raw vectors is also valid (but often messier visually).
    umap_embedding = reduce_dimensions_umap(vectors)
    
    # HDBSCAN Clustering
    cluster_labels = cluster_vectors_hdbscan(umap_embedding)
    
    # Build DataFrame
    df = pd.DataFrame({
        'word': words,
        'x': umap_embedding[:, 0],
        'y': umap_embedding[:, 1],
        'frequency': freqs,
        'cluster': cluster_labels,
        'log_freq': np.log1p(freqs)
    })
    
    # Analyze Clusters
    cluster_info = analyze_clusters(df)
    
    # Interactive Plot using Plotly
    # Points treated as noise by HDBSCAN are labeled -1
    df['cluster_name'] = df['cluster'].astype(str)
    df.loc[df['cluster'] == -1, 'cluster_name'] = 'Noise'
    
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='cluster_name',
        size='log_freq',
        hover_name='word',
        hover_data=['frequency', 'cluster'],
        title=f"Semantic Clusters: {model_name} (UMAP + HDBSCAN)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
    
    # Save to HTML (interactive)
    output_file = os.path.join(OUTPUT_DIR, f"{model_name.replace('.model', '')}_clusters.html")
    fig.write_html(output_file)
    print(f"Interactive plot saved to: {output_file}")

def main():
    # Define models to visualize
    # Ensure these match filenames in your models/ folder
    models_to_plot = [
        "republican_2021_2024.model",
        "democrats_2021_2024.model"
    ]
    
    for model_name in models_to_plot:
        try:
            visualize_embedding(model_name, min_count=100)
        except Exception as e:
            print(f"Failed to visualize {model_name}: {e}")

if __name__ == "__main__":
    main()