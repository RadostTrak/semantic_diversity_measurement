import numpy as np
import pandas as pd
from pathlib import Path
from vendi_score import vendi
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
n_seeds = 8
n_articles = [40, 60] # can't be smaller than total number of topics

# Load datasets
datasets = {
    'ag_news': Path('data/ag_news.csv'),
    'twenty_newsgroups': Path('data/twenty_newsgroups.csv')
}

# Load embedding files per dataset
embedding_files = {
    'ag_news': {
        'GloVe':  Path('embeddings/ag_news_glove.npy'),
        'MiniLM': Path('embeddings/ag_news_minilm.npy')
        # 'e5':   Path('embeddings/ag_news_e5.npy'),
    },
    'twenty_newsgroups': {
        'GloVe':  Path('embeddings/twenty_newsgroups_glove.npy'),
        'MiniLM': Path('embeddings/twenty_newsgroups_minilm.npy')
        # 'e5':   Path('embeddings/twenty_newsgroups_e5.npy'),
    },
}

# For saving results
def _row(dataset_name, k, model_name, number, base, seed, vs, mean_cos):
    return {
        'dataset': dataset_name,
        'k': k,
        'model': model_name,
        # 'label': label,   fix later
        'n_articles': number,
        'base_per_topic': base,
        'seed': seed,
        'vendi_score': float(vs),
        'mean_cosine': float(mean_cos),
    }

# Check to make sure all embedding files are the same length
for dataset_name, csv_path in datasets.items():
    df = pd.read_csv(csv_path)
    n_rows = len(df)
    print(f"\n=== {dataset_name} === ({n_rows} rows in CSV)")

    lengths = {'csv': n_rows}
    for embedding_name, npy_path in embedding_files[dataset_name].items():
        X = np.load(npy_path)
        lengths[embedding_name] = len(X)
        print(f"  {embedding_name}: {len(X)} rows")

# Apply GloVe mask and filter out embeddings with more than 256 tokens
masked_datasets = {}
masked_embeddings = {}

for dataset_name, csv_path in datasets.items():
    df = pd.read_csv(csv_path)
    keep_glove = (~df['glove_empty']).values
    keep_tokens = (df['minilm_tokens'] <= 256).values
    keep = keep_glove & keep_tokens
    masked_datasets[dataset_name] = df[keep].reset_index(drop=True)
    
    masked_embeddings[dataset_name] = {}
    for embedding_name, npy_path in embedding_files[dataset_name].items():
        X = np.load(npy_path)
        assert len(X) == len(keep), f"{dataset_name}/{embedding_name}: {len(X)} rows vs mask {len(keep)}"
        masked_embeddings[dataset_name][embedding_name] = X[keep]

    print(f"{dataset_name}: kept {keep.sum()} of {len(keep)} rows "
        f"(glove dropped {(~keep_glove).sum()}, >256 tokens dropped {(~keep_tokens).sum()})")


# Calculate Vendi score
results = []

for dataset_name, df in masked_datasets.items():
    print(f"\n=== {dataset_name} ===")

    # Obtain list of topics and number of topics
    topics = list(df['label_text'].unique())
    k_total = len(topics)

    for seed in range(n_seeds):
        topic_pools = {}

        # Shuffle topics
        topics_shuffled = list(np.random.default_rng(seed + 40_000).permutation(topics))

        # Shuffle all rows within topics
        for topic in topics:
            filtered_df = df[df['label_text'] == topic]
            shuffled = filtered_df.sample(frac=1, random_state=seed)
            topic_pools[topic] = shuffled.index.to_numpy() # store the shuffled indices

        # Get sample size per topic
        for number in n_articles:
            for k in range(1, k_total + 1):
                
                # Let each topic have base number of articles, distribute remainder
                base = number // k 
                remainder = number % k

                index_list = []
            
                for i, topic in enumerate(topics_shuffled[:k]):
                    pool = topic_pools[topic]

                    if i < remainder:
                        take = base + 1
                    else: 
                        take = base
                    assert len(pool) >= take, (
                        f"{dataset_name}/{topic}: pool has {len(pool)} rows "
                        f"but need {take} (k={k}, number={number})"
                    )
                    for idx in pool[:take]:
                        index_list.append(idx)
                index = np.array(index_list, dtype=int)

                # Compute vendi score and mean cosine similarity
                for model_name, embeddings in masked_embeddings[dataset_name].items():
                    X = embeddings[index]
                    vs = vendi.score_dual(X, normalize=True)

                    sim = cosine_similarity(X)
                    iu = np.triu_indices(len(X), k=1)
                    mean_cos = sim[iu].mean()

                    results.append(
                        _row(dataset_name, k, model_name, number, base, seed, vs, mean_cos)
                    )

results_df = pd.DataFrame(results)
expected_columns = list(_row('', 0, '', 0, 0, 0, 0.0, 0.0).keys())

assert list(results_df.columns) == expected_columns, (
    f"Column mismatch:\n  got:      {list(results_df.columns)}\n"
    f"  expected: {expected_columns}"
)

# Save results
out_path = Path('results/runs.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
model_order = ['GloVe', 'MiniLM']  # add e5 later
results_df['model'] = pd.Categorical(results_df['model'], categories=model_order, ordered=True)
results_df = results_df.sort_values('model', kind='stable').reset_index(drop=True)
results_df.to_csv(out_path, index=False)
print(f"\nWrote {len(results_df)} rows to {out_path}")