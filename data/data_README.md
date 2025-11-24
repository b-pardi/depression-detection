# Data Files

## DIAC-WOZ Dataset (Query Data)
**diac_woz_data.pkl**
- 189 patient conversations with embeddings
- Embeddings shape: (189, 384)
- Keys: 
  - `patient_ids`: Patient IDs
  - `conversations`: Full conversation text
  - `embeddings`: Pre-computed embeddings
  - `mdd_binary`: Binary MDD indicator (1/0)
  - `phq8_scores`: PHQ-8 scores (0-24)

## Clinical Knowledge Base (Reference Data)
**chunks.pkl**
- Text chunks from clinical depression literature
- Used as reference knowledge for RAG retrieval

**depression_embeddings.index** (FAISS)
- Embeddings for clinical chunks
- Used for similarity search to find top-k relevant chunks

**mandatory_context_DSM5_MDD.txt** ⚠️ REQUIRED FOR EVERY RAG CALL
- DSM-5 criteria for Major Depressive Disorder
- Must be included in EVERY prompt for consistent diagnosis
- Path: `data/RAG/mandatory_context_DSM5_MDD.txt`

**phq8.txt** ⚠️ REQUIRED FOR EVERY RAG CALL
- PHQ-8 questionnaire and scoring guide  
- Must be included in EVERY prompt for consistent diagnosis
- Path: `data/RAG/phq8.txt`

## Embedding Model
All embeddings created with: `SentenceTransformer('all-MiniLM-L6-v2')`
- Dimension: 384
- Fast inference, good quality for semantic search

## Usage Flow
1. Load `diac_woz_data.pkl` for patient conversations
2. Search `depression_embeddings.index` with patient embedding
3. Retrieve matching chunks from `chunks.pkl` using returned indices
4. Load mandatory context from `phq8.txt` and `mandatory_context_DSM5_MDD.txt`
5. Feed to LLM: patient convo + top-k chunks + mandatory context