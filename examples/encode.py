"""Text, code and an image in one vector space: rank three documents for a query.

    uv run --with "sentence-transformers[image]" --with torch examples/encode.py
"""
import torch
from sentence_transformers import SentenceTransformer

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer("zenlm/zen-embedding-2", model_kwargs={"torch_dtype": dtype},
                            config_kwargs={"audio_config": None})

query = model.encode("task: code retrieval | query: where is the cache key built?")
docs = [
    "title: cache.py | text: def make_key(args, kwargs): return hash((args, tuple(sorted(kwargs.items()))))",
    "title: http.py | text: def fetch(url): return requests.get(url).text",
    "title: none | text: Release notes for version 2.4.",
]
scores = model.similarity(query, model.encode(docs))[0]
for score, doc in sorted(zip(scores.tolist(), docs), reverse=True):
    print(f"{score:.3f}  {doc[:60]}")
