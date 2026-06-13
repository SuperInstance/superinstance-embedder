# SuperInstance Embedder

**SuperInstance Embedder** generates 32-dimensional embeddings from crate metadata, encoding each project's position in a 32-domain knowledge space. These "crate DNA" vectors seed Cloudflare Vectorize for semantic search across the SuperInstance ecosystem, enabling queries like "find crates related to ternary GPU computation."

## Why It Matters

With 100+ crates in the SuperInstance ecosystem, discovering relevant code requires more than keyword search — it requires semantic understanding. The embedder encodes each crate's identity across 32 orthogonal dimensions (ternary-math, agent-coordination, GPU compilation, crypto, distributed-systems, etc.), producing a compact fingerprint that captures cross-domain relationships. A crate at the intersection of "ternary-ml" and "agent-music" is semantically near both, even if its name contains neither word. These embeddings power the crate search engine, dependency recommendations, and fleet synergy detection — automatically identifying which crates could collaborate.

## How It Works

### Domain Space

The 32 dimensions represent orthogonal knowledge domains:

```
[ternary-math, ternary-ml, ternary-gpu, ternary-compression,
 agent-coordination, agent-music, agent-cognition, agent-timing,
 oxide-stack, cuda-compiler, character-building, education,
 compression, signal-processing, crypto, distributed,
 testing, formal-verification, creative-writing, physics,
 ecology, game-theory, scheduling, data-structure,
 compiler, runtime, iot, web,
 experimental, meta-cognition, scaling, synergy]
```

### Encoding Rules

The embedding is generated from crate metadata (name, description, domain tag, test count, LOC) using heuristic pattern-matching:

1. **Domain one-hot**: The declared domain sets that dimension to 1.0
2. **Cross-domain signals**: Name patterns trigger related dimensions:
   - `"ternary" + "kernel"` → ternary-math (0.7) + ternary-gpu (0.8)
   - `"agent" + "music"` → agent-music (0.8) + agent-cognition (0.5)
   - `"schedule" + "sync"` → agent-timing (0.8) + scheduling (0.5)
3. **Quality signal**: Test density maps to the testing dimension:
   ```
   testing_score = min(1.0, test_count / 30.0)
   ```
4. **LOC signal**: Code volume modulates the relevant domain dimensions

All values are clamped to [0, 1], producing a sparse vector with most dimensions at 0.

### Similarity Metric

Cosine similarity between crate embeddings identifies related projects:

```
similarity(A, B) = (A · B) / (||A|| · ||B||)
```

Complexity: O(d) = O(32) per comparison. For 1000 crates, brute-force search is O(1000 × 32) = 32K multiply-adds — sub-millisecond on any modern CPU. Cloudflare Vectorize accelerates this with ANN (Approximate Nearest Neighbor) indexing.

### Embedding Example

```rust
pub struct Embedding {
    pub name: String,
    pub vector: [f64; 32],
    pub metadata: CrateInfo,
}

// ternary-viterbi would produce:
// vector = [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//           0, 0.5, 0, 0, 0.3, 0, 0, 0,
//           0, 0.6, 0, 0, 0, 0, 0, 0,
//           0, 0.8, 0, 0]
// (ternary-math: 0.7, signal-processing: 0.5, testing: 0.3, game-theory: 0.6, meta-cognition: 0.8)
```

## Quick Start

```rust
use superinstance_embedder::{CrateInfo, Embedding};

fn main() {
    let info = CrateInfo {
        name: "ternary-viterbi".into(),
        tests: 12,
        loc: 350,
        domain: "ternary-math".into(),
        wave: 3,
        model: "glm".into(),
        description: "Viterbi decoder for ternary state sequences".into(),
    };

    let embedding = Embedding::from_crate(info);
    println!("Vector: {:?}", &embedding.vector[..8]);
    println!("Non-zero dimensions: {}",
        embedding.vector.iter().filter(|&&v| v > 0.0).count());
}
```

```bash
cargo build
cargo test
```

## API

| Type | Method | Description |
|------|--------|-------------|
| `CrateInfo` | — | Input metadata (name, tests, LOC, domain, wave) |
| `Embedding` | `from_crate(info)` | Generate 32-dim vector from metadata |
| `Embedding` | `vector: [f64; 32]` | The embedding itself |
| `DOMAINS` | `const [&str; 32]` | Domain labels |

## Architecture Notes

SuperInstance Embedder is the semantic indexing layer — it maps the fleet's output into a searchable knowledge space. Each crate's embedding captures its γ (constructive purpose: what it builds) and its cross-domain connections (synergy potential). The 32-dimensional space is coarse by design — it captures *relationships*, not fine-grained code semantics. Vectorize provides the ANN search that makes this practical at scale. In the γ + η = C framework, the embedder measures the diversity dimension of C: how broadly the fleet's competence spans. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NeurIPS*. — Word2vec: the inspiration for semantic embeddings.
2. Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." *IEEE Big Data*. — FAISS and ANN indexing.
3. Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation." *EMNLP*.

## License

MIT
