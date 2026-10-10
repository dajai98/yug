"""Memory subsystem: storage and retrieval of facts, external to the brain.

Per Architectural Decision #1 ("YUG is a system, not a model. The brain
is replaceable; memory, tools, permissions, verification live outside
it."), this package must not depend on any brain-specific type
(``Message``, ``GenerationResult``), any adapter
(``LlamaCppAdapter``), the native model (``YugTransformer``), or
tokenizer internals. Memory stores plain, JSON-serializable values.

Phase 2 scope only: working memory + basic JSON persistence +
retrieval. No embeddings, semantic search, vector storage, episodic-
memory architecture, or automatic memory management -- those belong to
later phases.
"""
