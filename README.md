# ⚖️ CLARA — Corpus Legal Andorrà amb Recuperació Augmentada

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/LangGraph-Agentic_RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Ollama-Local_Inference-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
  <img src="https://img.shields.io/badge/Llama_3.1-8B_Instruct-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Llama 3.1" />
  <img src="https://img.shields.io/badge/QLoRA-Unsloth_Fine--Tuning-FF6F00?style=for-the-badge" alt="QLoRA" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Storage-E25A1C?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Chainlit-Conversational_UI-F43F5E?style=for-the-badge" alt="Chainlit" />
  <a href="https://huggingface.co/MSS136/llama-clara-v2">
    <img src="https://img.shields.io/badge/Hugging%20Face-MSS136%2Fllama--clara--v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Model" />
  </a>
</p>

---

## 📌 Executive Summary

**CLARA** (*Corpus Legal Andorrà amb Recuperació Augmentada*) is an end-to-end autonomous Agentic Legal AI system tailored specifically to the legislation and legal framework of the **Principality of Andorra** (*Portal Jurídic del Principat d'Andorra* & *Butlletí Oficial del Principat d'Andorra - BOPA*).

Unlike generic RAG pipelines, CLARA integrates a **custom domain-fine-tuned Large Language Model** (`Llama-3.1-8B-Instruct` fine-tuned via QLoRA on Andorran legal instruction pairs) with an **Agentic LangGraph workflow** featuring **Hybrid Retrieval** (BM25 lexical + dense multilingual embeddings), **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **Self-Reflective Relevance Evaluation**.

All outputs are strictly grounded in retrieved statutory articles, delivered in standard legal Catalan, and formatted with Chain-of-Thought (`<think>`) reasoning steps and verified legal citations.

---

## 🏛️ System Architecture

```
[ Portal Jurídic d'Andorra ]
             │ (Web Scraping & Cleaning: status filter VIGENT / FITXA)
             ▼
    [ 01_Scraping ] ──> data/clean_texts/*.txt & metadata.jsonl
             │
             ▼
    [ 02_Indexar ] ──> Parent-Child Structural Legal Chunking (Articles, Capítols, Disposicions)
             │
      ┌──────┴──────────────────────────┐
      ▼                                 ▼
[ ChromaDB Vector Store ]       [ BM25 Lexical Index ]
(paraphrase-multilingual)       (Tokenized keywords)
      └──────┬──────────────────────────┘
             ▼
[ 04_FineTuning ] ──> Synthetic Q&A Generation -> QLoRA on Llama-3.1-8B -> GGUF Q4_K_M (Ollama: clara)
             │
             ▼
[ 03_Retrieving: LangGraph Agent ]
      │
      ├─► 1. REWRITE: User Question ──> Optimized Legal Catalan Search Keywords
      ├─► 2. DUAL RETRIEVE: BM25 (top 20) + ChromaDB (top 20)
      ├─► 3. RECIPROCAL RANK FUSION (RRF k=60): Merge & normalize candidates (top 20)
      ├─► 4. CROSS-ENCODER RERANK: BAAI/bge-reranker-v2-m3 (top 4 high-relevance chunks)
      ├─► 5. EVALUATE RELEVANCE: Pydantic Structured Self-Reflection
      └─► 6. GENERATE WITH THINKING: <think> CoT Deduction </think> + Final Catalan Answer
             │
             ▼
[ Chainlit Interactive UI ] ──> Real-time streaming, collapsible thought nodes, article citations
```

---

## 🚀 Key Technical Highlights

1. **Strict Legal Ingestion & Status Classification**:
   - Automated crawler for Tiki Wiki legal indexes classifying acts by legal validity: `VIGENT` (in force) and `FITXA` are retained; `DEROGAT` (repealed) and `VACATIO LEGIS` (pending) are systematically filtered out.
   - Text sanitation eliminates UI cruft, page numbering, promulgation formulas, and fixes broken Latin suffixes (*bis*, *ter*, *quater*).

2. **Parent-Child Hierarchical Legal Chunking**:
   - Preserves macro context (parent full law metadata) while splitting into actionable atomic units (`Article`, `Capítol`, `Disposició Transitòria / Final / Addicional`).
   - Validated against strict Pydantic schemas.

3. **Hybrid Retrieval with Reciprocal Rank Fusion & Cross-Encoder**:
   - **Lexical**: BM25 indexed over normalized Catalan tokens.
   - **Dense Semantic**: ChromaDB collection embedded with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   - **Fusion**: Reciprocal Rank Fusion ($k = 60$) blending both ranked candidate sets.
   - **Deep Reranking**: High-precision `BAAI/bge-reranker-v2-m3` cross-encoder scoring $(query, chunk)$ relevance pairs to select the top 4 definitive context windows.

4. **Domain Fine-Tuned Model (`clara` / `MSS136/llama-clara-v2`)**:
   - Base model: `Meta-Llama-3.1-8B-Instruct`.
   - Fine-tuned via **QLoRA (4-bit)** using **Unsloth** on an curated synthetic instruction dataset of Andorran legal questions, statutory deductions, and cross-references (`04_FineTuning/train_data.jsonl`).
   - Merged to 16-bit, quantized to `Q4_K_M` GGUF, and packaged for local CPU/GPU execution via Ollama.
   - Hugging Face repository: [MSS136/llama-clara-v2](https://huggingface.co/MSS136/llama-clara-v2).

5. **Agentic LangGraph Orchestration & Chainlit UI**:
   - Stateful workflow (`AgentState`) controlling query rewriting, retrieval, verification, and deductive generation.
   - Chainlit chat interface featuring real-time step streaming (`cl.Step`), separating hidden cognitive reasoning (`<think>`) from verified Catalan answers with clickable legal citations.

---

## 📂 Repository Structure

```
CLARA-Lleis-Andorra/
├── 01_Scraping/
│   ├── fase1_scraper_portalLleis.py    # Main scraper: downloads, classifies & cleans laws
│   └── scraper_cas_especial.py         # Dedicated scraper for edge-case annual compendiums
├── 02_Indexar/
│   ├── chunker.py                      # Hierarchical legal regex chunker (parent-child)
│   ├── indexer.py                      # Dual indexer: ChromaDB (vector) + BM25 (lexical)
│   └── fase2_exploracio_embeddings.ipynb # Jupyter notebook benchmarking embedding models
├── 03_Retrieving/
│   ├── agent.py                        # LangGraph agent graph (Rewrite -> Retrieve -> Rerank -> Eval -> Generate)
│   ├── app.py                          # Chainlit conversational interface with real-time step streaming
│   ├── chainlit.md                     # Welcome splash screen for Chainlit
│   └── retriever.py                    # LegalRetriever class (RRF + BAAI BGE Cross-Encoder)
├── 04_FineTuning/
│   ├── dataset_builder.py              # Synthetic QA dataset generator via local Ollama
│   ├── dataset_builder_groq.py         # Fast synthetic dataset generator via Groq API
│   ├── finetune_colab.ipynb            # Google Colab notebook: Unsloth QLoRA on Llama 3.1 8B
│   ├── convert_to_ollama.py            # Automation script: HuggingFace -> GGUF -> Ollama
│   ├── _dequant_bnb.py                 # Dequantization utility for bitsandbytes models
│   └── train_data.jsonl                # Fine-tuning dataset of instruction-response pairs
├── docs/
│   ├── CLARA_Document_Tecnic.docx      # Comprehensive technical architecture document
│   └── CLARA_Guia_Tecnica.docx         # Technical user manual & deployment guide
├── .chainlit/                          # Chainlit configuration & localization files
├── .gitattributes                      # Git attributes for LFS and line endings
├── .gitignore                          # Standard ignore rules (models, virtualenvs, cache, data)
├── Modelfile                           # Ollama Modelfile for the fine-tuned model (clara)
├── requirements.txt                    # Unified multi-stage dependencies
└── README.md                           # Master documentation
```

---

## 🛠️ Step-by-Step Setup & Execution

### 1. Prerequisites

- **Python**: 3.10 or 3.11 recommended.
- **Ollama**: Installed and running ([ollama.com](https://ollama.com)).
- **Hardware**:
  - Local inference with `Q4_K_M` GGUF: 8 GB+ RAM (or 6 GB+ VRAM with GPU acceleration).
  - Fine-tuning (optional): NVIDIA GPU with 12 GB+ VRAM (Google Colab T4 free tier is fully compatible).

### 2. Environment Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/marcs/CLARA-Lleis-Andorra.git
cd CLARA-Lleis-Andorra

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Setting Up the LLM in Ollama

You can run CLARA using the custom fine-tuned model or fall back to standard `qwen2.5:7b`.

#### Option A: Use the Custom Fine-Tuned Model (`clara`) [Recommended]
Download the fine-tuned GGUF from Hugging Face and register it in Ollama:

```bash
# 1. Download the quantized model from Hugging Face
huggingface-cli download MSS136/llama-clara-v2 llama-clara-v2-Q4_K_M.gguf --local-dir .

# 2. Build the Ollama model using the provided Modelfile
ollama create clara -f Modelfile

# 3. Test the model
ollama run clara "Quins són els requisits per obtenir la nacionalitat andorrana?"
```

#### Option B: Use Baseline Fallback Model
If running without the custom weights, pull `qwen2.5:7b`:

```bash
ollama pull qwen2.5:7b
```

---

### 4. Running the Pipeline

#### Phase 1: Scrape & Clean the Legal Corpus
```bash
cd 01_Scraping
python fase1_scraper_portalLleis.py
cd ..
```
*Outputs raw HTML and sanitized texts into `data/clean_texts/` along with `data/metadata.jsonl`.*

#### Phase 2: Structural Chunking & Dual Indexing
```bash
cd 02_Indexar
# Split into parent laws and atomic article chunks
python chunker.py

# Build ChromaDB vector collection and BM25 index
python indexer.py
cd ..
```
*Generates `data/chunks.jsonl`, `data/parent_store.json`, `data/chroma_db/`, and `data/bm25_index.pkl`.*

#### Phase 3: Domain Fine-Tuning (Optional / Advance Usage)
To train or adapt the model with custom instruction pairs:
- Use `04_FineTuning/dataset_builder.py` or `dataset_builder_groq.py` to generate QA training samples from `data/chunks.jsonl`.
- Open `04_FineTuning/finetune_colab.ipynb` in Google Colab (T4 GPU).
- Run the QLoRA fine-tuning routine with Unsloth.
- Convert and quantize the model using `04_FineTuning/convert_to_ollama.py`.

#### Phase 4: Launch the Interactive Chainlit Assistant
Launch the UI with hot-reloading:

```bash
cd 03_Retrieving
chainlit run app.py -w
```
Open your browser at `http://localhost:8000`.

To customize the Ollama model endpoints via environment variables:
```bash
# Example: Use clara for generation and qwen2.5:7b for fast deterministic rewriting
set OLLAMA_MODEL=clara
set OLLAMA_REWRITE_MODEL=qwen2.5:7b
set OLLAMA_EVALUATE_MODEL=qwen2.5:7b
chainlit run app.py -w
```

You can also run a direct CLI test of the agent graph:
```bash
python agent.py
```

---

## 🔍 LangGraph Execution Flow Details

When a query is entered into the system, the LangGraph engine coordinates the following operations:

| Node | Responsibility | Model / Engine |
|---|---|---|
| `REWRITE` | Normalizes user question into statutory Catalan keywords | `ChatOllama` (temp: `0.0`) |
| `RETRIEVE` | Queries both BM25 and ChromaDB, combines via RRF ($k=60$) | `rank_bm25` + `paraphrase-multilingual` |
| `RERANK` | Computes cross-attention relevance scores for top 20 candidates | `BAAI/bge-reranker-v2-m3` |
| `EVALUATE` | Verifies whether retrieved statutory articles answer the query | Structured Pydantic validation |
| `GENERATE` | Produces `<think>` reasoning block followed by cited answer in Catalan | `clara` / `Llama-3.1-8B` (temp: `0.2`) |

---

## 📚 Technical Documentation

Detailed architectural specifications, evaluation benchmarks, and implementation guides are available in the `docs/` directory:
- [CLARA Document Tècnic](docs/CLARA_Document_Tecnic.docx): Complete technical dossier covering scraping mechanics, chunking regexes, vector embeddings exploration, and QLoRA hyperparameter tuning.
- [CLARA Guia Tècnica](docs/CLARA_Guia_Tecnica.docx): Step-by-step administrator and operational deployment handbook.

---

## ⚖️ Legal Disclaimer

> **Important**: CLARA is an artificial intelligence research and assistance tool. Its responses are generated automatically for informational purposes and **do not constitute qualified legal advice**, statutory counsel, or legal representation. Users must verify all statutory articles and provisions against the official publications of the [Butlletí Oficial del Principat d'Andorra (BOPA)](https://www.bopa.ad) and consult a registered attorney (*advocat col·legiat al Col·legi d'Advocats d'Andorra*) for formal legal opinions.

---

## 📄 License & Attribution

- **Base Model**: Meta Llama 3.1 is licensed under the [Meta Llama 3.1 Community License](https://llama.meta.com/llama3_1/license/).
- **Project Source Code**: Distributed under the [MIT License](LICENSE).
- **Hugging Face Model Weights**: [MSS136/llama-clara-v2](https://huggingface.co/MSS136/llama-clara-v2).
