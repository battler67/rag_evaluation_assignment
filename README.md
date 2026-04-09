# Assignment 3: Custom RAG & Evaluation - Tardigrade Biology

This repository contains a full Retrieval-Augmented Generation (RAG) system with a custom evaluation framework centered on a niche domain: **The Biology and Survival Mechanisms of Tardigrades**.

#Demo video
<p align="center">
  <a href="https://www.youtube.com/watch?v=OGV1gEN6IDA">
    <img src="https://img.youtube.com/vi/OGV1gEN6IDA/maxresdefault.jpg" width="700">
  </a>
</p>

## Overview
- **Domain**: Biology and Survival Mechanisms of Tardigrades (Water Bears).
- **Dataset**: `[Simulated]` 6 comprehensive text documents and a 10-item Q&A evaluation dataset.
- **Goal**: Ingest documents, extract relevant chunks via semantic search, generate concise answers using an LLM, and evaluate the quality of those answers both quantitatively and qualitatively.

## RAG Pipeline Design
1. **Document Loading**: Uses `langchain` `DirectoryLoader`.
2. **Chunking Strategy**: `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=50`. This size ensures that discrete chunks encapsulate individual facts about tardigrades (e.g., Cryptobiosis mechanics) without losing context.
3. **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face. Very lightweight and provides strong semantic clustering.
4. **Vector Store**: `ChromaDB` for fast, local vector persistency.
5. **Generator Model**: `ChatOpenAI` configured in `.env`.

## Evaluation Framework Design
The evaluation process is executed over `data/qa_dataset.json` and produces `data/eval_results.json`. It comprises three components:

1. **Quantitative Metrics**:
   - *Semantic Similarity*: Measures Cosine Similarity between the embedding vectors of the generated answer and the pre-defined expected answer. Gives a broad indicator of semantic accuracy.
   - *Keyword Overlap*: Extracted via basic NLP techniques (removing stop words/punctuation). Indicates if specific crucial nouns are captured by the generator.

2. **Retrieval Performance Metric**:
   - Assesses what percentage of expected "key terms" from the expected answer appear directly inside the retrieved context chunks, indicating retrieval efficacy.

3. **Qualitative User Grading**:
   - A built-in feature in the Streamlit UI (Evaluation Tab). A human grader can review the Expected Answer, the Generated Answer, and score them on Coherence, Factual Correctness, and Completeness on a scale of 1-5.

## Installation & Setup
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. Set up the `.env` file with your OpenAI API key:
```env
OPENAI_API_KEY=sk-xxxx...
```
3. Run the application:
```bash
streamlit run app.py
```

## Discussion of Challenges
1. *Metric Scaling*: Cosine similarity using `all-MiniLM-L6-v2` generally groups closely clustered sentences in the `0.6 - 0.9` range. Differentiating a "perfect" vs "good" answer simply through a hard cosine threshold proved challenging without introducing more complex LLM-as-a-judge patterns. Simple keyword overlap provided a very strict grounding, but fails on synonyms. Ultimately, a combination of both metrics provided standard baseline signals.
2. *Hallucination risks*: Even specialized RAG can hallucinate if the LLM generalizes. The system prompt strongly commands it to "use concise, accurate sentences based ONLY on the context" to mitigate this.

