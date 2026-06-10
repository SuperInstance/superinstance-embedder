# superinstance-embedder

*32-dimensional DNA for every crate. Local embedder that seeds Cloudflare Vectorize.*

## What

Generates 32-dim embeddings from crate metadata (name, domain, test count, LOC, wave). Each dimension maps to a domain like "agent-music", "ternary-ml", "meta-cognition". The embedding captures what a crate IS, not just what it's called.

## Why

Cloudflare Vectorize needs vectors. This generates them from crate metadata without needing a neural network. The 32 dimensions are hand-crafted to capture the SuperInstance domain space:
- Dimensions 0-3: ternary (math, ML, GPU, compression)
- Dimensions 4-7: agent (coordination, music, cognition, timing)
- Dimensions 8-11: infrastructure (oxide, cuda, character, education)
- Dimensions 12-31: algorithms, quality, applications, systems, meta

## API

```rust
use superinstance_embedder::*;

let info = CrateInfo {
    name: "agent-sync".into(),
    tests: 10,
    loc: 1200,
    domain: "agent-timing".into(),
    wave: 65,
    model: "glm-5.1".into(),
    description: "T-minus timing protocol".into(),
};

let embedding = Embedding::from_crate(info);

// Find similar crates
let similar = index.find_similar(&embedding, 5);

// Find cross-domain synergies
let synergies = index.find_synergies("agent-music", 0.3, 5);

// Export for Vectorize
let json = index.to_vectorize_batch();
```

## Experiments

1. **Coverage analysis**: Which domains have the most crates?
2. **integration detection**: Cross-domain matches the authors missed
3. **Evolution tracking**: How embeddings change across waves
4. **Gap finding**: Empty dimensions = unexplored territory

## Part of the Fleet

Seeds `SuperInstance/superinstance-vectorize` (Cloudflare Worker) for edge-deployed semantic search.
