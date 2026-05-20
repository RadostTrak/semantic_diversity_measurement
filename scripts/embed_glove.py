import pandas as pd
import numpy as np
from pathlib import Path

# Read in data
df_ag = pd.read_csv('data/ag_news.csv')
df_20 = pd.read_csv('data/twenty_newsgroups.csv')

df_ag['label_text'] = df_ag['label'].map({0: 'world', 1: 'sports', 2: 'business', 3: 'sci_tech'})

# Load embedding vectors
expected_dim = 300
embeddings_index = {}
skipped = []

with open('pretrained/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt') as file:
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

# Tokenise each AG News article
article_tokens_ag = []
for text in df_ag['text']:
    tokens = []
    for word in text.lower().split():
        tokens.append(word.strip('.,!?";()[]{}'))
    article_tokens_ag.append(tokens)

# Mean pool per document
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
df_ag[['label', 'label_text']].to_csv('data/ag_news_meta.csv', index=False)
print(f"Saved {article_vectors_ag.shape} to embeddings/ag_news_glove.npy")

# Tokenise each Twenty Newsgroups article
article_tokens_20 = []
for text in df_20['text']:
    tokens = []
    for word in str(text).lower().split():
        tokens.append(word.strip('.,!?";()[]{}'))
    article_tokens_20.append(tokens)

# Mean pool per document
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
df_20[['label', 'label_text']].to_csv('data/twenty_newsgroups_meta.csv', index=False)
print(f"Saved {article_vectors_20.shape} to embeddings/twenty_newsgroups_glove.npy")