import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from vendi_score import vendi
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
dataset = 'ag_news'
meta_file = Path('data/ag_news_meta.csv')
results_file = Path('results/runs.csv')
n_docs_values = [100, 500, 1000, 5000, 10000, 30000]
n_topics_values = [1, 2, 3, 4]
n_seeds = 5
mean_cos_sample = 2000

embedding_models = {
    'glove':      Path('embeddings/ag_news_glove.npy'),
    'minilm':     Path('embeddings/ag_news_minilm.npy')
    # 'e5_mistral': Path('embeddings/ag_news_e5_mistral.npy')
}


def _mean_pairwise_cosine(X, sample_size, rng):
    """Mean off-diagonal cosine. Caps the matrix size when X has more rows than sample_size."""
    n = len(X)
    if n <= sample_size:
        K = cosine_similarity(X)
    else:
        idx = rng.choice(n, sample_size, replace=False)
        K = cosine_similarity(X[idx])
    upper = K[np.triu_indices_from(K, k=1)]
    return float(upper.mean())


def _row(condition, label, n_docs, seed, vs, mean_cos, run_id, embedding_name):
    return {
        'run_id':               run_id,
        'dataset':              dataset,
        'embedding':            embedding_name,
        'condition':            condition,
        'label':                label,
        'n_docs':               n_docs,
        'seed':                 seed,
        'vendi_score':          float(vs),
        'mean_pairwise_cosine': float(mean_cos),
    }


def per_topic_diversity(doc_vectors, meta, run_id, embedding_name):
    """Within-topic diversity at varying sample sizes."""
    results = []
    topics = sorted(meta['label_text'].unique().tolist())

    for topic in topics:
        mask = (meta['label_text'] == topic).values
        X_topic = doc_vectors[mask]
        n_available = len(X_topic)

        for n_docs in n_docs_values:
            if n_docs > n_available:
                continue
            seeds = [0] if n_docs == n_available else range(n_seeds)

            for seed in seeds:
                rng_doc = np.random.default_rng(seed)
                rng_cos = np.random.default_rng(seed + 10_000)

                idx = rng_doc.choice(n_available, n_docs, replace=False)
                X = X_topic[idx]

                vs = vendi.score_dual(X, normalize=True)
                mean_cos = _mean_pairwise_cosine(X, mean_cos_sample, rng_cos)

                results.append(_row('per_topic', topic, n_docs, seed, vs, mean_cos, run_id, embedding_name))
                print(f"[per_topic] {topic:>10s} n={n_docs:<6d} seed={seed} "
                      f"VS={vs:.3f} mean_cos={mean_cos:.3f}")
    return results


def n_topics_diversity(doc_vectors, meta, run_id, embedding_name, docs_per_topic=500):
    """Cross-topic diversity at varying k. Stratified: docs_per_topic from each of k topics."""
    results = []
    all_topics = sorted(meta['label_text'].unique().tolist())

    for k in n_topics_values:
        if k > len(all_topics):
            continue
        for seed in range(n_seeds):
            rng_doc = np.random.default_rng(seed)
            rng_cos = np.random.default_rng(seed + 20_000)

            chosen = list(rng_doc.choice(all_topics, k, replace=False))
            parts = []
            for t in chosen:
                mask = (meta['label_text'] == t).values
                X_t = doc_vectors[mask]
                take = min(docs_per_topic, len(X_t))
                pick = rng_doc.choice(len(X_t), take, replace=False)
                parts.append(X_t[pick])
            X = np.vstack(parts)

            vs = vendi.score_dual(X, normalize=True)
            mean_cos = _mean_pairwise_cosine(X, mean_cos_sample, rng_cos)

            label = f"k={k}:[{','.join(sorted(chosen))}]"
            results.append(_row('n_topics', label, len(X), seed, vs, mean_cos, run_id, embedding_name))
            print(f"[n_topics] k={k} seed={seed} topics={sorted(chosen)} "
                  f"VS={vs:.3f} mean_cos={mean_cos:.3f}")
    return results


# Run
meta = pd.read_csv(meta_file)
results_file.parent.mkdir(parents=True, exist_ok=True)
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

all_results = []
for embedding_name, embeddings_file in embedding_models.items():
    print(f"\n=== {embedding_name} ===")
    doc_vectors = np.load(embeddings_file)
    assert len(doc_vectors) == len(meta), f"{embedding_name}: embeddings and meta must align row-wise"
    all_results += per_topic_diversity(doc_vectors, meta, run_id, embedding_name)
    all_results += n_topics_diversity(doc_vectors, meta, run_id, embedding_name)

df_out = pd.DataFrame(all_results)
df_out.to_csv(results_file, mode='a', header=not results_file.exists(), index=False)
print(f"\nSaved {len(all_results)} rows to {results_file} (run_id={run_id})")