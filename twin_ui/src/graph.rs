use std::sync::{Arc, RwLock};
use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};

/// A document node in the wikilink knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocNode {
    pub doc_id: String,
    pub title: String,
    pub source_path: String,
}

/// Normalized link frequency between two documents (0.0–1.0).
pub type EdgeWeight = f32;

/// Thread-safe knowledge graph built from Obsidian wikilink metadata.
/// Wrapped in Arc<RwLock<...>> so the background loader and UI thread can share it.
pub type KnowledgeGraph = Arc<RwLock<DiGraph<DocNode, EdgeWeight>>>;

/// Create an empty knowledge graph ready for background population.
pub fn new_graph() -> KnowledgeGraph {
    Arc::new(RwLock::new(DiGraph::new()))
}

/// Graph proximity score for combined retrieval formula.
/// 1 hop → 1.0, 2 hops → 0.5, beyond → 0.0.
pub fn proximity_score(hops: u32) -> f32 {
    match hops {
        1 => 1.0,
        2 => 0.5,
        _ => 0.0,
    }
}

/// Combined retrieval score: weighted sum of vector similarity and graph proximity.
/// Weights default to TWIN_GRAPH_VECTOR_WEIGHT (0.7) and TWIN_GRAPH_PROXIMITY_WEIGHT (0.3).
pub fn combined_score(vector_sim: f32, graph_prox: f32, vector_w: f32, graph_w: f32) -> f32 {
    (vector_w * vector_sim) + (graph_w * graph_prox)
}
