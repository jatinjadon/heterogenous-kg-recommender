# heterogenous-kg-recommender
# 🎬 Graph Attention Network (GATv2) Recommendation Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jatinjadon/heterogenous-kg-recommender/blob/main/GATv2_recommendation_model.ipynb)

## 📌 Overview
This repository contains an end-to-end recommendation engine built using **Heterogeneous Graph Neural Networks (GNNs)**. By modeling users, movies, actors, and genres as a complex Knowledge Graph, this system leverages a Multi-Head Graph Attention Network (GATv2) to perform highly accurate link prediction. 

This project serves as a comprehensive exploration of applied network analysis, demonstrating how structural topology and multi-modal node relationships can solve the classic "cold-start" problem found in traditional collaborative filtering systems.

## 🚀 Key Features
* **Heterogeneous Graph Architecture:** Models intricate metadata (`directed`, `acted_in`, `has_genre`) alongside `user-movie` ratings to maintain a dense web of collaborative signals.
* **Multi-Head Attention (GATv2):** The encoder dynamically learns the mathematical importance of different edge types during message passing, funneling high-dimensional node embeddings into a dot-product edge decoder.
* **Robust Data Engineering:** Implements strict K-Core filtering to mathematically prevent user isolation prior to executing PyTorch Geometric's random link splitting and negative sampling.
* **Deterministic Pipeline:** Configured with strict seed locks across Python, NumPy, and PyTorch CUDA backends to ensure 100% reproducible training loops.
* **Production Inference:** Vectorized matrix multiplication allows for instantaneous `HitRate@K` evaluation and human-readable movie recommendations on the fly.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
* **Data Processing:** Pandas, NumPy
* **Evaluation:** Scikit-learn (ROC-AUC), Custom HitRate@10 Metric Engine

## 📊 Datasets
The notebook's initial execution cell automates the entire ETL pipeline, fetching, extracting, and merging the following data on the fly:
1. **MovieLens (Small):** 100,000 ratings acting as the primary interaction edges.
2. **TMDB 5000:** Rich metadata (Cast, Crew, Genres) acting as the structural anchor nodes. *(Streamed directly via compressed GitHub hosting to bypass Kaggle API authentication walls).*

## 💻 How to Run (One-Click Deploy)
You do not need to manually download any CSVs or configure a local GPU environment to test this engine. 

1. Click the **Open in Colab** badge at the top of this page.
2. In the Google Colab menu bar, click `Runtime` -> `Run all`.
3. The environment will automatically provision a cloud GPU, stream the datasets, map the Knowledge Graph, train the multi-head GATv2 model, and output final predictions.
