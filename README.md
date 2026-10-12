---
license: apache-2.0
base_model: google/embeddinggemma-2
library_name: sentence-transformers
pipeline_tag: feature-extraction
language:
- multilingual
tags:
- zen
- zen-embedding
- zenlm
- hanzo
- embedding
- sentence-transformers
- multimodal-embedding
- vision
- audio
- video
- retrieval
- matryoshka
---

# Zen Embedding 2

One embedding space for text (including code), images, video and audio: any mix of them in, a
768-dimensional vector out, so a text query retrieves a screenshot, a clip or a code span directly.
8,192-token context (minutes of audio or video), Matryoshka truncation to 512, 256 or 128 dimensions,
and encoders that load per modality: 270M parameters for text alone, 740M with vision and audio.

Repackaged from [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0,
Google DeepMind) at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, weights unchanged. **Not trained
from scratch.** The upstream card has the benchmarks, training data and safety notes.

## Usage

```python
import torch
from sentence_transformers import SentenceTransformer

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer("zenlm/zen-embedding-2", model_kwargs={"torch_dtype": dtype})

query = model.encode("task: code retrieval | query: where is the cache key built?")
doc = model.encode("title: cache.py | text: def make_key(args, kwargs): ...")
print(model.similarity(query, doc))
```

Run it in `bfloat16` or `float32`, never `float16`: its activations overflow `float16` into NaN.

Task prefixes steer the vector. Queries take `task: search result | query: …`,
`task: question answering | query: …`, `task: fact checking | query: …` or
`task: code retrieval | query: …`; documents take `title: {title or none} | text: …`; symmetric
tasks put `task: classification | query: …`, `task: clustering | query: …` or
`task: sentence similarity | query: …` on every input.

Mixed input marks each item's place with `<|image|>`, `<|video|>` or `<|audio|>`:

```python
emb = model.encode({
    "text": "Checkout page after submit: <|image|> Screen recording of the failure: <|video|>",
    "image": ["checkout.png"],
    "video": "failure.mp4",
})
```

Text only, without the vision and audio encoders:

```python
model = SentenceTransformer("zenlm/zen-embedding-2",
                            config_kwargs={"vision_config": None, "audio_config": None})
```

Truncate with `model.encode(x, truncate_dim=256, normalize_embeddings=True)`; queries and documents
must share a dimension.

Served on the Hanzo gateway (`api.hanzo.ai`) as `zen-embedding-2`.

## License

`apache-2.0`. Upstream: **google/embeddinggemma-2** (EmbeddingGemma 2) by Google DeepMind. This repository
redistributes the weights unchanged under the same license.
