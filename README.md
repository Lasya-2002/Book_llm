# 📚 NextChapter - Book Recommendation Based on Large Language Models

An AI-powered conversational book recommendation system built using Large Language Models (LLMs), Sentence Transformers, FAISS vector similarity search, and Falcon-1B-Instruct fine-tuning.

---

## 🌟 Overview

With the exponential growth of digital books and online reading platforms, discovering books that align with individual interests has become increasingly difficult.

This project presents an intelligent recommendation system that combines:

- Semantic search
- Transformer-based embeddings
- Conversational AI
- Large Language Model fine-tuning

The system allows users to discover books through natural language interaction while generating highly relevant and context-aware recommendations.

---

## 🎯 Problem Statement

Traditional recommendation systems often rely on:

- genre matching
- keyword overlap
- rating similarity
- metadata filtering

These methods frequently produce:

- repetitive suggestions
- popularity bias
- poor contextual understanding
- limited personalization

This project addresses those limitations by building an AI-driven recommendation chatbot capable of understanding semantic meaning and user intent using modern transformer architectures.

---

# 🧠 Objectives

The primary objectives of this project were:

- Build an intelligent conversational book recommendation system
- Generate semantically meaningful recommendations
- Improve personalization using transformer embeddings
- Fine-tune an instruction-following LLM for recommendation generation
- Create an interactive user experience using conversational AI
- Compare traditional recommendation techniques with modern LLM-based approaches

---

# 🏗️ System Architecture

The final system combines:

1. Sentence Transformer Embeddings
2. FAISS Similarity Search
3. Falcon-1B-Instruct Fine-Tuned Model
4. Flask Backend API
5. Streamlit Frontend Interface

### Workflow

```text
User Query
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Similarity Search
    ↓
Context Retrieval
    ↓
Falcon-1B-Instruct Generation
    ↓
Personalized Book Recommendations
```

---

# 📂 Dataset

## Source

Dataset collected using the Google Books API.

## Features Used

The dataset includes:

- Book titles
- Authors
- Categories
- Descriptions
- Ratings
- Metadata

## Final Dataset Size

After:

- missing value treatment
- duplicate removal
- preprocessing

Final dataset dimensions:

- **296,816 rows**
- **10 columns**

---

# ⚙️ Methodology

## 1. Data Preparation

The dataset was cleaned and standardized to ensure consistency across:

- titles
- authors
- descriptions
- categories
- ratings

This improved embedding quality and recommendation relevance.

---

## 2. Embedding Generation

Sentence Transformers were used to generate semantic embeddings for textual features.

### Model Used

```text
all-MiniLM-L6-v2
```

Generated embeddings were stored using:

```text
FAISS IndexFlatL2
```

This enabled fast and scalable semantic similarity search.

---

## 3. Instruction Dataset Construction

A large instruction-style dataset was generated for LLM fine-tuning.

### Process

- Multiple recommendation prompts were generated
- Instruction-response pairs were constructed
- JSONL format dataset was created

### Final Instruction Dataset Size

Approximately:

```text
2,305,139 entries
```

---

## 4. Fine-Tuning Falcon-1B-Instruct

The Falcon model was fine-tuned using:

- Hugging Face Transformers
- PyTorch
- Accelerate library
- H100 GPU on Ola Krutrim Cloud

---

# 🤖 Models Used

## Sentence Transformer

### Configuration

| Parameter | Value |
|---|---|
| Model | all-MiniLM-L6-v2 |
| Base Architecture | MiniLM |
| Transformer Layers | 6 |
| Hidden Size | 384 |
| Pooling | Mean Pooling |
| Training Objective | Contrastive Learning |

### Purpose

Used for:

- semantic embedding generation
- similarity search
- recommendation retrieval
- instruction dataset generation

---

## Falcon-1B-Instruct

### Configuration

| Parameter | Value |
|---|---|
| Model | tiiuae/Falcon-1B-Instruct |
| Parameters | ~1.3 Billion |
| Architecture | Decoder-only Transformer |
| Transformer Layers | 24 |
| Hidden Size | 2048 |
| Attention Heads | 32 |
| Context Length | 2048 tokens |
| Training Objective | Causal Language Modeling |

### Purpose

Used for:

- conversational recommendation generation
- instruction following
- context-aware responses
- personalized recommendations

---

# 🧮 Mathematical Concepts Used

## Sentence Transformer

Core concepts include:

- Positional Encoding
- Multi-Head Self Attention
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Mean Pooling

### Embedding Pipeline

```text
Input Tokens
    ↓
Token Embeddings + Positional Encoding
    ↓
Transformer Layers
    ↓
Mean Pooling
    ↓
Sentence Embedding
```

---

## Falcon-1B

The Falcon model uses:

- Decoder-only Transformer Architecture
- Causal Self Attention
- Residual Connections
- Feedforward Networks
- Instruction Fine-Tuning

### Pipeline

```text
Input Prompt
    ↓
Token Embedding
    ↓
Causal Attention Blocks
    ↓
Language Generation
    ↓
Recommendation Response
```

---

# 🏋️ Model Training

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 1 |
| Batch Size | 32 |
| Effective Batch Size | 64 |
| Learning Rate | 0.00005 |
| Optimizer | AdamW |
| Precision | bfloat16 |
| Logging Steps | 20000 |
| Saving Steps | 50000 |

---

## Training Infrastructure

Fine-tuning was performed using:

```text
Ola Krutrim Cloud
NVIDIA H100 GPU
```

---

# 📈 Results

## Model Performance

| Model | Observation |
|---|---|
| Bag of Words | Poor semantic understanding |
| Bag of N-Grams | High complexity and repetitive suggestions |
| TF-IDF | Loss of contextual meaning |
| Word2Vec | Weak long-description understanding |
| BERT | Good semantic recommendations but lacks generative ability |
| Falcon-1B | Strong semantic understanding with conversational generation |

---

## Final Metrics

| Metric | Value |
|---|---|
| Training Loss | 0.2 |
| Evaluation Loss | 0.15 |
| Perplexity | 5.15 |

The Falcon-1B-based system generated highly relevant and semantically meaningful recommendations while supporting conversational interaction.

---

# 💻 Deployment

## Backend

Built using:

```text
Flask
```

Responsibilities:

- API handling
- model inference
- recommendation generation
- vector retrieval

---

## Frontend

Built using:

```text
Streamlit
```

Features:

- conversational interface
- filters
- recommendation exploration
- interactive user experience

---

# 🚀 Features

- Conversational recommendation system
- Semantic similarity search
- LLM-based recommendation generation
- Transformer embeddings
- FAISS vector indexing
- Personalized recommendations
- Instruction-tuned LLM
- Streamlit interactive frontend
- Flask backend API

---

# 📊 Comparative Analysis

The project compared multiple recommendation approaches:

- Bag of Words
- N-Grams
- TF-IDF
- Word2Vec
- BERT
- Falcon-1B-Instruct

The Falcon-based system significantly outperformed traditional techniques due to:

- contextual understanding
- semantic depth
- instruction following
- generative capability

---

# 🔮 Future Improvements

## Hybrid Recommendation Systems

- Combine collaborative filtering with LLMs
- Knowledge graph integration
- Reinforcement learning-based personalization

---

## Real-Time Personalization

- Dynamic user feedback loops
- Session-aware recommendations
- Adaptive recommendation pipelines

---

## Edge Optimization

- Model distillation
- Lightweight inference pipelines
- Offline recommendation support

---

# 🛠️ Tech Stack

## Languages & Frameworks

- Python
- Flask
- Streamlit
- PyTorch
- Hugging Face Transformers

## AI & NLP

- Sentence Transformers
- Falcon-1B-Instruct
- FAISS

## Infrastructure

- Ola Krutrim Cloud
- NVIDIA H100 GPU

---

# 📦 Installation

You can clone the repository from github

## Run Backend

```bash
python app.py
```

---

## Run Streamlit Frontend

```bash
streamlit run frontend.py
```

---

# 🤝 Contributors

- M. V. Sri Lasya
- M. Sai Dheeraj

Supervisor:

- Dr. Motahar Reza

---

# 📚 References

Key references include:

- Attention Is All You Need (Vaswani et al.)
- Research on LLM-based recommender systems
- Falcon architecture studies
- Transformer recommendation system literature
- Semantic recommendation system research

---

# 🧠 Key Learnings

This project provided practical experience in:

- Large Language Models
- Transformer architectures
- Recommendation systems
- Semantic search
- Vector databases
- Fine-tuning workflows
- Instruction tuning
- GPU-based training pipelines
- Conversational AI systems

---

# ✨ Conclusion

This project demonstrates how modern Large Language Models can significantly improve recommendation quality by understanding deeper semantic relationships and conversational context.

By combining:

- Sentence Transformers
- FAISS retrieval
- Falcon-1B fine-tuning
- conversational interfaces

we developed a scalable and intelligent recommendation system capable of delivering highly personalized and context-aware book suggestions.

The results highlight the growing importance of LLMs in next generation recommender system and is a real time case study on existing advanced recommendation systems.
