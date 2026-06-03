import pandas as pd
import numpy as np
from pathlib import Path

# Read in data
df_ag = pd.read_csv("data/ag_news.csv")
df_20 = pd.read_csv("data/twenty_newsgroups.csv")

# Load embedding vectors
expected_dim = 300
embeddings_index = {}
skipped = []

with open("pretrained/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt") as file:
    for line in file:
        values = line.split()
        if len(values) != expected_dim + 1:
            if values:
                skipped.append(values[0])
            continue
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            skipped.append(word)

print(f"Loaded {len(embeddings_index)} vectors, skipped {len(skipped)}")

# Tokenise AG News articles
article_tokens_ag = []
for text in df_ag['text']:
    tokens = []
    for word in str(text).lower().split():
        tokens.append(word.strip('.,!?";()[]{}'))
    article_tokens_ag.append(tokens)

# Average token vectors per article to obtain article-level embeddings
article_vectors_ag = []
for tokens in article_tokens_ag:
    in_vocab = []
    for word in tokens:
        if word in embeddings_index:
            in_vocab.append(embeddings_index[word])
    if in_vocab:
        article_vectors_ag.append(np.mean(in_vocab, axis=0))
    else:
        article_vectors_ag.append(np.zeros(300, dtype='float32'))

article_vectors_ag = np.vstack(article_vectors_ag)
Path('embeddings').mkdir(exist_ok=True)
np.save('embeddings/ag_news_glove.npy', article_vectors_ag)
print(f"Saved {article_vectors_ag.shape} to embeddings/ag_news_glove.npy")

# Tokenise Twenty Newsgroups articles
article_tokens_20 = []
for text in df_20['text']:
    tokens = []
    for word in str(text).lower().split():
        tokens.append(word.strip('.,!?";()[]{}'))
    article_tokens_20.append(tokens)

# Average token vectors per article to obtain article-level embeddings
article_vectors_20 = []
for tokens in article_tokens_20:
    in_vocab = []
    for word in tokens:
        if word in embeddings_index:
            in_vocab.append(embeddings_index[word])
    if in_vocab:
        article_vectors_20.append(np.mean(in_vocab, axis=0))
    else:
        article_vectors_20.append(np.zeros(300, dtype='float32'))

article_vectors_20 = np.vstack(article_vectors_20)
np.save('embeddings/twenty_newsgroups_glove.npy', article_vectors_20)
print(f"Saved {article_vectors_20.shape} to embeddings/twenty_newsgroups_glove.npy")

# Check for all-zero vectors (articles with no in-vocab tokens)
n_empty_ag = sum(1 for v in article_vectors_ag if not v.any())
print(f"AG News no in-vocab tokens (all-zero vectors): {n_empty_ag}") # 0

n_empty_20 = sum(1 for v in article_vectors_20 if not v.any())
print(f"Twenty Newsgroups no in-vocab tokens (all-zero vectors): {n_empty_20}") # 20

# Check which articles have zero vectors in Twenty Newsgroups
glove_empty_20 = ~article_vectors_20.any(axis=1)
print(f"Zero-vector rows: {glove_empty_20.sum()}")
print(df_20[glove_empty_20][['label_text', 'text']].to_string()) # non-semantic punctuation and URLs

# Create new column in dataframes to be able to filter out all-zero vectors in analysis
df_20['glove_empty'] = glove_empty_20
df_20.to_csv('data/twenty_newsgroups.csv', index=False)

glove_empty_ag = ~article_vectors_ag.any(axis=1)
df_ag['glove_empty'] = glove_empty_ag
df_ag.to_csv('data/ag_news.csv', index=False)