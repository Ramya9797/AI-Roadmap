# Customer Intelligence RAG

This module provides the knowledge retrieval layer
for the Customer Intelligence AI Platform.

Pipeline:

Documents
    ↓
Loader
    ↓
Chunker
    ↓
Metadata
    ↓
Embeddings
    ↓
Chroma
    ↓
Retriever
    ↓
LLM
    ↓
Grounded Answer
    ↓
Citations