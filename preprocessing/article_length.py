import pandas as pd
import numpy as np
from transformers import AutoTokenizer

ag_news = pd.read_csv('data/ag_news.csv')
twenty_newsgroups = pd.read_csv('data/twenty_newsgroups.csv')

minilm_limit = 256

# Tokenize articles
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

ag_news_length = []
twenty_newsgroups_length = []

for article in ag_news['text']:
    tokens = tokenizer.encode(str(article), add_special_tokens=True, truncation=False)
    ag_news_length.append(len(tokens))

for article in twenty_newsgroups['text']:
    tokens = tokenizer.encode(str(article), add_special_tokens=True, truncation=False)
    twenty_newsgroups_length.append(len(tokens))

# Obtain descriptive statistics for AG News
print("AG News article length statistics:")
print(f"Mean: {np.mean(ag_news_length):.2f} tokens")
print(f"Median: {np.median(ag_news_length)} tokens")
print(f"Max: {np.max(ag_news_length)} tokens")
print(f"Over {minilm_limit} tokens: {(np.array(ag_news_length) > minilm_limit).sum()} articles")
print(f"Under {minilm_limit} tokens: {(np.array(ag_news_length) < minilm_limit).sum()} articles")
print(f"Percentage of articles over {minilm_limit} tokens: {(np.array(ag_news_length) > minilm_limit).mean() * 100:.2f}%")

# Obtain per-topic descriptive statistics for AG News
ag_news['n_tokens'] = ag_news_length

per_topic = ag_news.groupby('label')['n_tokens'].agg(
    over=lambda s: (s > minilm_limit).sum(),
    percent_over=lambda s: (s > minilm_limit).mean() * 100,
)
print("AG News per-topic statistics:")
print(per_topic)

# Obtain descriptive statistics for 20 Newsgroups
print("\n20 Newsgroups article length statistics:")
print(f"Mean: {np.mean(twenty_newsgroups_length):.2f} tokens")
print(f"Median: {np.median(twenty_newsgroups_length)} tokens")
print(f"Max: {np.max(twenty_newsgroups_length)} tokens")
print(f"Over {minilm_limit} tokens: {(np.array(twenty_newsgroups_length) > minilm_limit).sum()} articles")
print(f"Under {minilm_limit} tokens: {(np.array(twenty_newsgroups_length) <= minilm_limit).sum()} articles")
print(f"Percentage of articles over {minilm_limit} tokens: {(np.array(twenty_newsgroups_length) > minilm_limit).mean() * 100:.2f}%")

# Obtain per-topic descriptive statistics for 20 Newsgroups
twenty_newsgroups['n_tokens'] = twenty_newsgroups_length

per_topic = twenty_newsgroups.groupby('label_text')['n_tokens'].agg(
    over=lambda s: (s > minilm_limit).sum(),
    percent_over=lambda s: (s > minilm_limit).mean() * 100,
)
print("20 Newsgroups per-topic statistics:")
print(per_topic)