# Semantic Drift & Community Divergence in Political Subreddits

> 🚧 **Work in Progress**: This project is currently under active development. New features and analysis modules are being added regularly.
> _Draft Reference: [Different-Word-Usage-on-Different-Subreddits](https://github.com/Lucas265163/Different-Word-Usage-on-Different-Subreddits/tree/main)_

A comprehensive NLP pipeline analyzing the evolution of language on Reddit. This project quantifies how political polarization manifests in semantic shifts using temporal word embeddings.

## Key Features

1.  **Memory-Efficient Preprocessing**: 
    - Handles massive Reddit dumps (`.zst` compression) using streaming and chunking.
    - Custom spaCy pipeline for cleaning, lemmatization, and phrase detection.

2.  **Temporal Word Embeddings (Word2Vec)**:
    - Trains separate vector spaces for distinct political eras (e.g., *Before 2016*, *2017-2020*, *2021-2024*).
    - Detects compound terms (e.g., "White_House", "Climate_Change") using a global Bigram model.

3.  **Semantic Alignment (Procrustes Analysis)**:
    - Aligns vector spaces between different communities (e.g., Democrats vs. Republicans) using Orthogonal Procrustes Analysis.
    - Enables direct mathematical comparison of how different groups use the same words.

4.  **Polarization Metrics**:
    - **Distance Analysis**: Measures the cosine distance of shared vocabulary between communities.
    - **Axis Analysis**: Projects words onto an ideological axis (e.g., *Conservative <-> Liberal*) to quantify their partisan lean.

5.  **Interactive Visualization**:
    - Dimensionality reduction via **UMAP**.
    - Density-based clustering via **HDBSCAN** to automatically find semantic topics.

## Project Structure

```text
├── data/
│   ├── raw/                # Place .zst or .jsonl files here
│   └── preprocessed/       # Generated pickle chunks
├── models/                 # Trained Word2Vec and Bigram models
├── output/                 # CSV reports and HTML visualizations
├── src/
│   ├── preprocessing.py    # Data cleaning & chunking
│   ├── training.py         # Word2Vec & Phraser training
│   ├── distance_analysis.py# Cosine distance & Procrustes alignment
│   ├── axis_analysis.py    # Ideological axis projection
│   ├── visualization.py    # UMAP + HDBSCAN + Plotly
│   └── file_streams.py     # Zstandard stream handlers
└── main.py                 # Unified CLI entry point
