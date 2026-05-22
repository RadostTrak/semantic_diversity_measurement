import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "intfloat/e5-mistral-7b-instruct",
    trust_remote_code=True,
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16},
)

print(f"Model loaded. GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

instruction = "Instruct: Identify the topic of the given news article\nQuery: "

sample_texts = [
    "The stock market reached a new high today amid strong earnings reports.",
    "Investors are optimistic about tech sector growth this quarter.",
    "The basketball team won the championship after a dramatic final game.",
    "The athlete broke the world record in the 100 meter sprint.",
    "Scientists discovered a new species of deep sea fish.",
    "Researchers published findings on a novel protein folding mechanism.",
]

embeddings = model.encode(sample_texts, prompt=instruction)
print(f"Shape: {embeddings.shape}")

# Sanity check: within-topic similarity should be higher than between-topic
sim = cosine_similarity(embeddings)
print("\nPairwise cosine similarity matrix:")
print(np.round(sim, 3))

print("\nExpected pattern: pairs (0,1), (2,3), (4,5) should have higher similarity than other pairs.")
print(f"Finance pair (0,1):    {sim[0, 1]:.3f}")
print(f"Sports pair (2,3):     {sim[2, 3]:.3f}")
print(f"Science pair (4,5):    {sim[4, 5]:.3f}")
print(f"Cross-topic mean:      {np.mean([sim[0,2], sim[0,4], sim[2,4]]):.3f}")

np.save("embeddings_e5_test.npy", embeddings)
print("\nSaved embeddings.")