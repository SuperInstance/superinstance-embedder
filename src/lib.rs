//! # superinstance-embedder
//!
//! Generate 32-dimensional embeddings from crate source code.
//! Seeds Cloudflare Vectorize with crate "DNA" for semantic search.

/// The 32 domain dimensions
pub const DOMAINS: [&str; 32] = [
    "ternary-math", "ternary-ml", "ternary-gpu", "ternary-compression",
    "agent-coordination", "agent-music", "agent-cognition", "agent-timing",
    "oxide-stack", "cuda-compiler", "character-building", "education",
    "compression", "signal-processing", "crypto", "distributed",
    "testing", "formal-verification", "creative-writing", "physics",
    "ecology", "game-theory", "scheduling", "data-structure",
    "compiler", "runtime", "iot", "web",
    "experimental", "meta-cognition", "scaling", "synergy",
];

/// Crate metadata for embedding
#[derive(Debug, Clone)]
pub struct CrateInfo {
    pub name: String,
    pub tests: u32,
    pub loc: u32,
    pub domain: String,
    pub wave: u32,
    pub model: String,
    pub description: String,
}

/// A 32-dimensional embedding
#[derive(Debug, Clone)]
pub struct Embedding {
    pub name: String,
    pub vector: [f64; 32],
    pub metadata: CrateInfo,
}

impl Embedding {
    /// Generate embedding from crate info
    pub fn from_crate(info: CrateInfo) -> Self {
        let mut vec = [0.0f64; 32];
        
        // Domain encoding
        for (i, domain) in DOMAINS.iter().enumerate() {
            if info.domain == *domain {
                vec[i] = 1.0;
            }
        }
        
        // Pattern-based cross-domain signals
        let name = info.name.to_lowercase();
        
        // Music-cognition crossover
        if name.contains("agent") && (name.contains("music") || name.contains("jam") || 
            name.contains("riff") || name.contains("groove") || name.contains("soul") ||
            name.contains("sync") || name.contains("swing") || name.contains("counterpoint") ||
            name.contains("ensemble") || name.contains("orchestration") || name.contains("intonation") ||
            name.contains("phrasing") || name.contains("transcription") || name.contains("resonance") ||
            name.contains("polyrhythm") || name.contains("overtone") || name.contains("cadence") ||
            name.contains("staccato") || name.contains("legato") || name.contains("rubato") ||
            name.contains("fermata") || name.contains("anacrusis") || name.contains("motif") ||
            name.contains("harmonic") || name.contains("contrapuntal") || name.contains("choir") ||
            name.contains("venue") || name.contains("audience") || name.contains("call-response") ||
            name.contains("microtone")) {
            vec[5] = (vec[5] + 0.8).min(1.0); // agent-music
            vec[6] = (vec[6] + 0.5).min(1.0); // agent-cognition
            vec[7] = (vec[7] + 0.3).min(1.0); // agent-timing
        }
        
        // Ternary foundation
        if name.contains("ternary") {
            vec[0] = (vec[0] + 0.7).min(1.0);
            if name.contains("matmul") || name.contains("conv") || name.contains("transformer") {
                vec[1] = (vec[1] + 0.8).min(1.0); // ternary-ml
            }
            if name.contains("kernel") || name.contains("warp") || name.contains("grid") {
                vec[2] = (vec[2] + 0.8).min(1.0); // ternary-gpu
            }
            if name.contains("compress") || name.contains("encode") || name.contains("quantize") {
                vec[3] = (vec[3] + 0.7).min(1.0); // ternary-compression
            }
        }
        
        // Timing/scheduling
        if name.contains("schedule") || name.contains("sync") || name.contains("timing") || 
            name.contains("gate") || name.contains("rubato") || name.contains("fermata") {
            vec[7] = (vec[7] + 0.8).min(1.0); // agent-timing
            vec[22] = (vec[22] + 0.5).min(1.0); // scheduling
        }
        
        // Meta-cognition
        if name.contains("self-rivalry") || name.contains("metamorphosis") || 
            name.contains("phase-change") || name.contains("speciation") ||
            name.contains("dream") || name.contains("semiosis") {
            vec[29] = (vec[29] + 0.8).min(1.0); // meta-cognition
            vec[31] = (vec[31] + 0.6).min(1.0); // synergy
        }
        
        // Test density
        vec[16] = (info.tests as f64 / 30.0).min(1.0);
        
        // LOC density
        vec[25] = (info.loc as f64 / 10000.0).min(1.0);
        
        // Wave recency
        vec[30] = (info.wave as f64 / 70.0).min(1.0);
        
        // Normalize
        let mag = vec.iter().map(|v| v * v).sum::<f64>().sqrt().max(0.0001);
        for v in vec.iter_mut() {
            *v /= mag;
        }
        
        Embedding { name: info.name.clone(), vector: vec, metadata: info }
    }
    
    /// Cosine similarity between two embeddings
    pub fn cosine_similarity(&self, other: &Embedding) -> f64 {
        let dot: f64 = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a * b).sum();
        let mag_a: f64 = self.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mag_b: f64 = other.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 { return 0.0; }
        dot / (mag_a * mag_b)
    }
    
    /// Is this a cross-domain match? (different primary domain but high similarity)
    pub fn is_cross_domain_synergy(&self, other: &Embedding, threshold: f64) -> bool {
        let similarity = self.cosine_similarity(other);
        // Different primary domain but similar enough
        let self_primary = self.primary_domain();
        let other_primary = other.primary_domain();
        self_primary != other_primary && similarity >= threshold
    }
    
    /// Primary domain (highest dimension)
    pub fn primary_domain(&self) -> &str {
        let max_idx = self.vector.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        DOMAINS[max_idx]
    }
    
    /// Export as JSON for Vectorize ingestion
    pub fn to_vectorize_json(&self) -> String {
        let values: Vec<f64> = self.vector.to_vec();
        format!(
            r#"{{"id":"{}","values":[{}],"metadata":{{"name":"{}","tests":{},"domain":"{}","wave":{}}}}}"#,
            self.name,
            values.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(","),
            self.metadata.name,
            self.metadata.tests,
            self.metadata.domain,
            self.metadata.wave,
        )
    }
}

/// A collection of embeddings for batch operations
pub struct EmbeddingIndex {
    pub embeddings: Vec<Embedding>,
}

impl EmbeddingIndex {
    pub fn new() -> Self {
        Self { embeddings: Vec::new() }
    }
    
    pub fn add(&mut self, info: CrateInfo) {
        self.embeddings.push(Embedding::from_crate(info));
    }
    
    /// Find top-K most similar crates
    pub fn find_similar(&self, query: &Embedding, top_k: usize) -> Vec<(&Embedding, f64)> {
        let mut scored: Vec<_> = self.embeddings.iter()
            .filter(|e| e.name != query.name)
            .map(|e| (e, query.cosine_similarity(e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).collect()
    }
    
    /// Find cross-domain synergies
    pub fn find_synergies(&self, domain: &str, threshold: f64, top_k: usize) -> Vec<(&Embedding, f64)> {
        let domain_crates: Vec<_> = self.embeddings.iter()
            .filter(|e| e.metadata.domain == domain)
            .collect();
        
        if domain_crates.is_empty() { return Vec::new(); }
        
        // For each domain crate, find cross-domain matches
        let mut synergies: Vec<(&Embedding, f64)> = Vec::new();
        
        for dc in &domain_crates {
            for other in &self.embeddings {
                if other.metadata.domain != domain {
                    let sim = dc.cosine_similarity(other);
                    if sim >= threshold {
                        synergies.push((other, sim));
                    }
                }
            }
        }
        
        // Deduplicate by name and sort
        let mut seen = std::collections::HashSet::new();
        synergies.retain(|(e, _)| seen.insert(&e.name));
        synergies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        synergies.into_iter().take(top_k).collect()
    }
    
    /// Export all embeddings as Vectorize JSON array
    pub fn to_vectorize_batch(&self) -> String {
        let items: Vec<String> = self.embeddings.iter().map(|e| e.to_vectorize_json()).collect();
        format!("[{}]", items.join(","))
    }
    
    /// Coverage analysis: which domains have the most crates?
    pub fn domain_coverage(&self) -> Vec<(&str, usize)> {
        let mut counts = std::collections::HashMap::new();
        for e in &self.embeddings {
            let domain = e.primary_domain();
            *counts.entry(domain).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_crate(name: &str, domain: &str) -> CrateInfo {
        CrateInfo {
            name: name.to_string(),
            tests: 20,
            loc: 5000,
            domain: domain.to_string(),
            wave: 69,
            model: "glm-5.1".to_string(),
            description: "test".to_string(),
        }
    }

    #[test]
    fn test_embedding_creation() {
        let info = test_crate("agent-sync", "agent-timing");
        let emb = Embedding::from_crate(info);
        assert_eq!(emb.name, "agent-sync");
        let mag: f64 = emb.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((mag - 1.0).abs() < 0.01, "should be normalized");
    }

    #[test]
    fn test_cosine_similarity_self() {
        let info = test_crate("agent-sync", "agent-timing");
        let emb = Embedding::from_crate(info);
        let sim = emb.cosine_similarity(&emb);
        assert!((sim - 1.0).abs() < 0.01, "self similarity should be 1.0");
    }

    #[test]
    fn test_cosine_similarity_different() {
        let info_a = test_crate("ternary-matmul", "ternary-ml");
        let info_b = test_crate("agent-sync", "agent-timing");
        let emb_a = Embedding::from_crate(info_a);
        let emb_b = Embedding::from_crate(info_b);
        let sim = emb_a.cosine_similarity(&emb_b);
        assert!(sim < 0.9, "different domains should have lower similarity, got {sim}");
    }

    #[test]
    fn test_primary_domain() {
        let info = test_crate("ternary-matmul", "ternary-ml");
        let emb = Embedding::from_crate(info);
        assert_eq!(emb.primary_domain(), "ternary-ml");
    }

    #[test]
    fn test_cross_domain_synergy() {
        let info_a = test_crate("agent-sync", "agent-timing");
        let info_b = test_crate("ternary-scheduler", "ternary-math");
        let emb_a = Embedding::from_crate(info_a);
        let emb_b = Embedding::from_crate(info_b);
        // These are different domains but both involve timing/scheduling
        let is_synergy = emb_a.is_cross_domain_synergy(&emb_b, 0.0);
        assert!(is_synergy, "timing crates should be synergistic");
    }

    #[test]
    fn test_find_similar() {
        let mut index = EmbeddingIndex::new();
        index.add(test_crate("agent-sync", "agent-timing"));
        index.add(test_crate("agent-groove", "agent-music"));
        index.add(test_crate("ternary-matmul", "ternary-ml"));
        
        let query = Embedding::from_crate(test_crate("agent-swing", "agent-timing"));
        let similar = index.find_similar(&query, 2);
        assert_eq!(similar.len(), 2);
        // agent-sync should be more similar than ternary-matmul
        assert!(similar[0].1 > similar[1].1);
    }

    #[test]
    fn test_find_synergies() {
        let mut index = EmbeddingIndex::new();
        index.add(test_crate("agent-jam", "agent-music"));
        index.add(test_crate("agent-sync", "agent-timing"));
        index.add(test_crate("ternary-scheduler", "ternary-math"));
        
        let synergies = index.find_synergies("agent-music", 0.0, 5);
        // Should find cross-domain matches
        assert!(!synergies.is_empty());
    }

    #[test]
    fn test_domain_coverage() {
        let mut index = EmbeddingIndex::new();
        index.add(test_crate("a", "agent-music"));
        index.add(test_crate("b", "agent-music"));
        index.add(test_crate("c", "ternary-ml"));
        
        let coverage = index.domain_coverage();
        assert!(!coverage.is_empty());
        assert_eq!(coverage[0].1, 2); // agent-music has 2
    }

    #[test]
    fn test_vectorize_json_export() {
        let info = test_crate("agent-sync", "agent-timing");
        let emb = Embedding::from_crate(info);
        let json = emb.to_vectorize_json();
        assert!(json.contains("agent-sync"));
        assert!(json.contains("values"));
        assert!(json.contains("metadata"));
    }

    #[test]
    fn test_batch_export() {
        let mut index = EmbeddingIndex::new();
        index.add(test_crate("a", "agent-music"));
        index.add(test_crate("b", "ternary-ml"));
        let batch = index.to_vectorize_batch();
        assert!(batch.starts_with('['));
        assert!(batch.ends_with(']'));
    }

    #[test]
    fn test_music_crate_detection() {
        let info = test_crate("agent-jam", "agent-music");
        let emb = Embedding::from_crate(info);
        // Should have high values in music and cognition dimensions
        assert!(emb.vector[5] > 0.0, "should detect music dimension");
    }
}
