# AI Eval & RLHF Vector Observatory 3D

An interactive WebGL2 / GLSL 3D observability studio visualizing how modern LLM Retrieval-Augmented Generation (RAG), HNSW vector indexing, hallucination drift, RLHF/DPO reward steering, and adversarial safety guardrails operate inside high-dimensional latent embedding space.

## Key Engineering Highlights
- **3,200-Vector Latent Embedding Manifold**: Custom GLSL point-cloud shaders with depth attenuation, dynamic Top-K highlighting, and zero-stutter spatial raycast inspection across 5 semantic domains.
- **Authentic 3-Layer HNSW Graph Search**: Real greedy multi-layer traversal (`Layer 2 Express Hubs -> Layer 1 Regional Bridges -> Layer 0 Semantic Neighborhood`) computing live Precision@K and NDCG@K.
- **Live Geometric AI Evaluation Engine**: Faithfulness, Semantic Relevance, Safety Margin, and KL-Divergence are derived directly from vector geometry, grounding hulls, and trajectory steering shifts.
- **4 Interactive Failure & Alignment Modes**:
  1. **Grounded RAG Pipeline**
  2. **Hallucination & Semantic Drift** (adjust RLHF β live to pull ungrounded answers back into the evidence hull)
  3. **RLHF / DPO Reward Beam Steering** (6 parallel candidate trajectories scored & pruned in real time)
  4. **Adversarial Jailbreak Guardrail** (dynamic Safety Hyperplane shockwave & trajectory deflection)
- **Zero-Obstruction Studio HUD**: Independently collapsible left/right docks (`[` / `]`) and full Zen Mode (`H`).
