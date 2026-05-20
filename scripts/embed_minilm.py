import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Read in data
df_ag = pd.read_csv('data/ag_news.csv')
df_20 = pd.read_csv('data/twenty_newsgroups.csv')
df_ag['label_text'] = df_ag['label'].map({0: 'world', 1: 'sports', 2: 'business', 3: 'sci_tech'})

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Embed AG News articles
embeddings_ag = model.encode(df_ag['text'].tolist(), batch_size=64, show_progress_bar=True)
print(embeddings_ag[0].shape)  # Should be (384,) for all-MiniLM-L6-v2

Path('embeddings').mkdir(exist_ok=True)
np.save('embeddings/ag_news_minilm.npy', embeddings_ag)
print(f"Saved {embeddings_ag.shape} to embeddings/ag_news_minilm.npy")

# Embed Twenty Newsgroups articles
embeddings_20 = model.encode(df_20['text'].tolist(), batch_size=64, show_progress_bar=True)

Path('embeddings').mkdir(exist_ok=True)
np.save('embeddings/twenty_newsgroups_minilm.npy', embeddings_20)
print(f"Saved {embeddings_20.shape} to embeddings/twenty_newsgroups_minilm.npy")