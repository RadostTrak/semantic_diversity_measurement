import pandas as pd
import numpy as np
from collections import Counter
from vendi_score import vendi

# Read data
df = pd.read_csv('data/ag_news.csv')

# Filter for topic 1 (Sport) only
filtered_df = df[df['label']==1]

# Load embedding vectors
expected_dim = 300
embeddings_index = {}
skipped = []

with open('wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt') as file:
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

# Tokenise each news article
words = []

for text in filtered_df['text']:
    text = text.lower().split()
    for word in text:
        words.append(word.strip('.,!?";()[]{}'))

word_counts = Counter(words)
print(f"Unique words: {len(word_counts)}")

# Another strategy
docs_tokens = []

for text in filtered_df['text']:
    tokens = []
    text = text.lower().split()
    for word in text:
        tokens.append(word.strip('.,!?";()[]{}'))
    docs_tokens.append(tokens)

# Match vocabulary to embeddings
oov = []
for w in word_counts:
    if w not in embeddings_index:
        oov.append(w)

print(f"Out-of-vocabulary words: {len(oov)}, sample: {oov[:10]}")

# Mean pool per document
doc_vectors = []
for tokens in docs_tokens:
    in_vocab = []
    for word in tokens:
        if word in embeddings_index:
            in_vocab.append(embeddings_index[word])
    
    if in_vocab:
        doc_vectors.append(np.mean(in_vocab, axis=0))
    else:
        doc_vectors.append(np.zeros(300, dtype='float32'))

doc_vectors = np.vstack(doc_vectors)
print(doc_vectors.shape)

# Calculate Vendi score
vs_cosine = vendi.score_dual(doc_vectors, normalize=True)
print(f"Vendi Score: {vs_cosine:}")