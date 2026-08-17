# CLARA — Corpus Legal Andorrà amb Recuperació Augmentada

CLARA is a Retrieval-Augmented Generation (RAG) system for Andorran legislation. It scrapes the *Portal Jurídic del Principat d'Andorra*, builds a hybrid semantic + lexical search index over the legal corpus, and exposes it through an agentic pipeline with a Chainlit chat UI. Answers are grounded in retrieved legal fragments, generated in Catalan, and always cite the specific articles used.

## How it works

The project is organized as a three-stage pipeline, mirrored by the folder structure:

```
Clara/
├── 01_Scraping/       Fase 1 — download & clean the legal corpus
├── 02_Indexar/        Fase 2 — chunk the corpus and build the search indexes
└── 03_Retrieving/     Fase 3 — hybrid retrieval, agent graph, and chat app
```

### 1. Scraping (`01_Scraping/`)

- **`fase1_scraper_portalLleis.py`** — Main scraper. Crawls `IndexPerAnys` on the Portal Jurídic (a Tiki Wiki site), year by year, and classifies each law by status using its status icon:
  - ✅ `VIGENT` (in force) — included
  - ✅ `FITXA` (record with no further tracking) — included, configurable via `INCLOU_FITXES`
  - ❌ `VACATIO LEGIS` (approved, not yet in force) — excluded
  - ❌ `DEROGAT` (repealed) — excluded

  For each included law it downloads the detail page, extracts and cleans the text (removes Tiki Wiki UI cruft, page numbers, administrative boilerplate/signatures, repairs line-broken Latin suffixes like *bis/ter/quater*), and writes:
  - `data/raw_html/<year|fitxes>/*.html` — cached raw pages
  - `data/clean_texts/<year|fitxes>/*.txt` — cleaned plain text
  - `data/metadata.jsonl` — one JSON record per law (title, year, status, URL, paths)
  - `data/scraper.log` — run log

  Configurable at the top of the file: `ANYS_A_SCRAPEJAR` (years to scrape, empty = all detected), `INCLOU_FITXES`, `MAX_DOCS_TOTAL` (cap for quick tests), and request rate limits.

- **`scraper_cas_especial.py`** — Variant of the same scraper for special/edge cases in year or case indexing on the portal.

### 2. Indexing (`02_Indexar/`)

- **`chunker.py`** — Reads `data/clean_texts/*.txt` and splits each law into a two-level hierarchy:
  - **Parent document**: the full law (`data/parent_store.json`)
  - **Child chunks**: individual `Article`, `Capítol`, or `Disposició` units, each linked to its parent via `parent_id` (`data/chunks.jsonl`)

  Uses regex tuned for Catalan legal structure (including addicional/transitòria/final dispositions) and validates records with Pydantic.

- **`indexer.py`** — Builds a **dual index** over the chunks:
  - **Semantic (vector)**: ChromaDB collection `lleis_andorra`, embedded with the multilingual model `paraphrase-multilingual-MiniLM-L12-v2` → persisted to `data/chroma_db/`
  - **Lexical (keyword)**: BM25 via `rank_bm25`, tokenized (lowercased, punctuation-stripped) → persisted to `data/bm25_index.pkl`

  Re-run this script any time `data/chunks.jsonl` changes; it rebuilds the Chroma collection from scratch.

### 3. Retrieval & Agent (`03_Retrieving/`)

- **`retriever.py`** — `LegalRetriever` class implementing a two-stage retrieve pipeline:
  1. BM25 search → top 20 candidates
  2. ChromaDB semantic search → top 20 candidates
  3. **Reciprocal Rank Fusion** (k=60) merges both ranked lists into 20 unique candidates
  4. **Cross-encoder re-ranking** (`BAAI/bge-reranker-v2-m3`) scores each (query, chunk) pair and returns the top 4 final fragments

- **`agent.py`** — LangGraph agent (`AgentState` → `StateGraph`) that orchestrates the full answer flow using a local **Qwen2.5:7b** model served via Ollama:

  ```
  REWRITE → RETRIEVE_AND_RERANK → EVALUATE_RELEVANCE → GENERATE_WITH_THINKING → END
  ```

  - **rewrite**: turns the user's natural-language question into an optimized Catalan keyword query for hybrid search
  - **retrieve_and_rerank**: calls `LegalRetriever.cercar()`
  - **evaluate_relevance**: LLM judges (structured/Pydantic output) whether retrieved fragments are sufficient to answer; a conditional fallback branch (reformulate / expand retrieval) is scaffolded but not wired into the linear flow
  - **generate_with_thinking**: forces the model to emit a `<think>...</think>` reasoning block (facts → applicable articles → deductive reasoning) followed by the final answer, always in Catalan and citing specific articles, never inventing law not present in the retrieved fragments

- **`app.py`** — [Chainlit](https://chainlit.io) chat interface. On session start it loads the retriever and compiles the agent graph once (`@cl.on_chat_start`). On each user message (`@cl.on_message`) it streams the graph's execution, showing each node (rewrite, retrieve, evaluate, generate) as a collapsible `cl.Step`, splits the `<think>` reasoning block from the final Catalan answer in real time, and attaches the actually-retrieved legal chunks as source citations on the final message.

- **`agent copy.py`** — working/backup copy of `agent.py`.

## Requirements

- Python 3.11
- Local [Ollama](https://ollama.com) instance running the `qwen2.5:7b` model (`ollama pull qwen2.5:7b`)
- Python packages:
  ```
  requests
  beautifulsoup4
  pydantic
  chromadb
  rank_bm25
  sentence-transformers
  langchain-core
  langchain-ollama
  langgraph
  chainlit
  ```

## Usage

Run each phase in order from the corresponding folder (each stage reads/writes a shared `data/` directory):

```bash
# 1. Scrape and clean the corpus
cd 01_Scraping
python fase1_scraper_portalLleis.py

# 2. Chunk and index
cd ../02_Indexar
python chunker.py
python indexer.py

# 3. Launch the chat app
cd ../03_Retrieving
chainlit run app.py -w
```

You can also run the agent directly from the command line without the UI for a quick end-to-end test:

```bash
cd 03_Retrieving
python agent.py
```

## Notes

- All scraped/cleaned text, chunks, and indexes live under a shared `data/` folder (created automatically) relative to wherever the scripts are run — keep this consistent across the three stages, or point later stages at the right paths.
- The `__pycache__/` folder under `03_Retrieving/` is a build artifact and can be safely deleted.
- Source language throughout the corpus and generated answers is Catalan; some internal code comments/docstrings are in Spanish.
