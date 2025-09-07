# RAG (Retrieval-Augmented Generation) Implementation

A Python implementation task to build a complete RAG system for microwave manual assistance using PostgreSQL with pgvector extension and OpenAI API

## 🎓 Learning Goals

By completing this task, you will learn:
- How to implement the complete RAG pipeline: **Retrieval**, **Augmentation**, and **Generation**
- How to work with vector embeddings and similarity search using PostgreSQL with pgvector
- How to process and chunk text documents for vector storage
- How to perform semantic search with cosine and Euclidean distance metrics
- Understanding RAG architecture without high-level frameworks

## 📋 Requirements

- Python 3.11+
- Docker and Docker Compose
- pip

## 🔧 Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**

3. **Start PostgreSQL with pgvector:**
   ```bash
   docker-compose up -d
   ```
   This will start PostgreSQL on port 5433 with the pgvector extension enabled.

4. **Project structure:**
   ```
   task/
   ├── app.py                           🚧 TODO - Main RAG application
   ├── embeddings/
   │   ├── embeddings_client.py         🚧 TODO - Embeddings client
   │   ├── text_processor.py            🚧 TODO - Text processing and vector operations
   │   └── microwave_manual.txt         ✅ Knowledge base document
   ├── chat/
   │   └── chat_completion_client.py    ✅ Сhat completion client
   ├── models/
   │   ├── conversation.py              ✅ Conversation management
   │   ├── message.py                   ✅ Message structure
   │   └── role.py                      ✅ Role definitions
   └── utils/
       └── text.py                      ✅ Text chunking utilities
   docker-compose.yml                   ✅ PostgreSQL with pgvector
   init-scripts/init.sql                ✅ Database initialization
   ```

## 🖌️ Application Architecture:

The RAG system follows a three-step process:
1. **🔍 Retrieval**: Find relevant chunks from the microwave manual using vector similarity search
2. **🔗 Augmentation**: Combine retrieved context with user question in a structured prompt
3. **🤖 Generation**: Use LLM to generate accurate answer based on the provided context

## 📝 Your Tasks

### If the task in the main branch is hard for you, then switch to the `with-detailed-description` branch

Complete the implementation by filling in all the TODO sections across these files:

### 🔌 **Step 1: Embeddings Client (`embeddings_client.py`)**
- Complete the `get_embeddings()` method to call embeddings API
- Parse the response and extract embeddings data
- Handle the request/response format according to API specification

### 📊 **Step 2: Text Processing (`text_processor.py`)**
- Implement `process_text_file()` to load, chunk, and store document embeddings
- Complete `_truncate_table()` for database cleanup
- Implement `_save_chunk()` to store text chunks with embeddings in PostgreSQL
- Complete `search()` method for semantic similarity search using pgvector

### 🚀 **Step 3: Main Application (`app.py`)**
- Initialize clients with proper model deployments
- Implement document processing workflow
- Complete the RAG pipeline: Retrieval → Augmentation → Generation
- Handle user interaction and conversation management

### 🔥 **Step 4: Additional task**
There are 2 key questions you need to investigate:
1. What happens when the database contains embeddings with 384 dimensions, but you search using an embedding with 385 dimensions?
2. What happens when you use `text-embedding-3-small` for context generation but `voyage-3.5` for search (both have 384 dimensions)?

Investigate both scenarios in your project implementation. To do that you need to implement EmbeddingClient for Anthropic `voyage-3.5` embedding model. https://docs.anthropic.com/en/docs/build-with-claude/embeddings#voyage-http-api



## 🔧 Implementation Details

### Database Schema
The PostgreSQL database uses the pgvector extension with this schema:
```sql
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    document_name VARCHAR(64),
    text TEXT NOT NULL,
    embedding VECTOR(384)
);
```

### Similarity Search
The system supports two distance metrics:
- **Cosine Distance** (`<=>` operator): Measures angle between vectors
- **Euclidean Distance** (`<->` operator): Measures straight-line distance

### Configuration Parameters
Experiment with these parameters for optimal performance:
- `chunk_size`: Size of text chunks (default: 150, recommended: 300)
- `overlap`: Character overlap between chunks (default: 40)
- `top_k`: Number of relevant chunks to retrieve (default: 5)
- `min_score`: Similarity threshold (range: 0.1-0.99, default: 0.5)
- `dimensions`: Embedding dimensions (1536 for OpenAI models)

## 🎯 Testing Your Implementation

### Valid request samples:
``` 
What safety precautions should be taken to avoid exposure to excessive microwave energy?
```
```
What is the maximum cooking time that can be set on the DW 395 HCG microwave oven?
```
```
How should you clean the glass tray of the microwave oven?
```
```
What materials are safe to use in this microwave during both microwave and grill cooking modes?
```
```
What are the steps to set the clock time on the DW 395 HCG microwave oven?
```
```
What is the ECO function on this microwave and how do you activate it?
```
```
What are the specifications for proper installation, including the required free space around the oven?
```
```
How does the multi-stage cooking feature work, and what types of cooking programs cannot be included in it?
```
```
What should you do if food in plastic or paper containers starts smoking during heating?
```
```
What is the recommended procedure for removing odors from the microwave oven?
```

### Invalid request samples:
```
What do you know about the Codeus community?
```
```
What do you think about the dinosaur era? Why did they die?
```
