# Semantic Search Engine with Sentence Transformers and Vector Embeddings

This project demonstrates how to build a **semantic search engine** using pre-trained transformer models with Hugging Face's `Sentence Transformers` library and vector embeddings. 
Unlike traditional keyword-based search, semantic search understands the meaning behind queries and retrieves the most relevant results based on the context.

<!--## Key Features
- **BERT-based Embeddings**: Leverages pre-trained BERT model to generate contextual embeddings for text.
- **Vector Similarity Search**: Uses cosine similarity to find the most semantically similar documents or sentences.
- **Scalable and Efficient**: Designed to handle large datasets with efficient vector search techniques.
- **Easy to Use**: Simple API for indexing and querying documents.-->

## How it works?
Input text (documents or queries) is converted into dense vector representations (also known as vector embeddings) using a transformer model. These vector embeddings are then stored in a vector database for efficient retrieval. When a query is entered during search time, its embedding is compared with indexed document embeddings using cosine similarity metric and the most semantically similar documents are returned as search results.
