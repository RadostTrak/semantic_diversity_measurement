import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

# Batch size 2 is chosen to fit the model in GPU memory
batch_size = 2
max_seq_length = 4096

instruction = "Instruct: Identify the topic or theme of the given text\nQuery: "

# Load the datasets and add labels for AG News
df_ag = pd.read_csv('data/ag_news.csv')
df_20 = pd.read_csv('data/twenty_newsgroups.csv')
df_ag['label_text'] = df_ag['label'].map({0: 'world', 1: 'sports', 2: 'business', 3: 'sci_tech'})

# Load the model
print("Loading E5-Mistral-7B-Instruct...")
model = SentenceTransformer(
    'intfloat/e5-mistral-7b-instruct',
    trust_remote_code=True,
    device='cuda',
    model_kwargs={'torch_dtype': torch.float16},
)

model.max_seq_length = max_seq_length
print(f"Model loaded. GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

Path('embeddings').mkdir(exist_ok=True)

# Embed AG News
print(f"\nEmbedding AG News ({len(df_ag)} documents, batch_size={batch_size})...")
start = time.time()
embeddings_ag = model.encode(
    df_ag['text'].tolist(),
    prompt=instruction,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
)

elapsed = time.time() - start
print(f"AG News done in {elapsed/60:.1f} min. Shape: {embeddings_ag.shape}")

np.save('embeddings/ag_news_e5.npy', embeddings_ag)
print("Saved to embeddings/ag_news_e5.npy")

# Clear cached GPU memory before the next dataset
torch.cuda.empty_cache()

# Embed Twenty Newsgroups
print(f"\nEmbedding Twenty Newsgroups ({len(df_20)} documents, batch_size={batch_size})...")
start = time.time()
embeddings_20 = model.encode(
    df_20['text'].tolist(),
    prompt=instruction,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
)

elapsed = time.time() - start
print(f"Twenty Newsgroups done in {elapsed/60:.1f} min. Shape: {embeddings_20.shape}")

# Save embeddings
np.save('embeddings/twenty_newsgroups_e5.npy', embeddings_20)
print("Saved to embeddings/twenty_newsgroups_e5.npy")

print("\nAll embeddings complete.")