"""
src/indexer.py
===============

Mòdul 2 de la Fase 1 del sistema RAG per a textos legals d'Andorra.

Carrega els chunks generats per `src/chunker.py` (`data/chunks.jsonl`) i
construeix una **indexació doble**:

    1. Índex SEMÀNTIC (vectorial) amb ChromaDB, utilitzant l'embedder
       multilingüe `paraphrase-multilingual-MiniLM-L12-v2` (obligatori
       perquè els textos són en català).
    2. Índex LÈXIC (BM25) amb rank_bm25, per a cerca per coincidència
       exacta de termes (clau en textos legals, on la terminologia
       precisa —p.ex. "disposició addicional segona"— sol ser decisiva).

Sortides:
    - data/chroma_db/            -> base de dades vectorial persistent
                                     (col·lecció 'lleis_andorra', recreada
                                     de zero a cada execució)
    - data/bm25_index.pkl        -> {bm25_index, chunk_ids} serialitzat

Autor: Enginyeria de Dades RAG
"""

from __future__ import annotations

import json
import pickle
import re
import string
from pathlib import Path
from typing import List

import chromadb
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# CONFIGURACIÓ GLOBAL
# ---------------------------------------------------------------------------

CHUNKS_INPUT_PATH = Path("data/chunks.jsonl")
CHROMA_PERSIST_DIR = Path("data/chroma_db")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")

COLLECTION_NAME = "lleis_andorra"

# Model multilingüe obligatori: bon rendiment en català per a cerca
# semàntica de textos legals.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Mida del lot per generar embeddings i inserir a ChromaDB.
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# MODEL DE DADES (Pydantic) — valida cada chunk carregat des del JSONL
# ---------------------------------------------------------------------------

class ChildChunk(BaseModel):
    """Representa un chunk 'Fill' tal com es va persistir a chunks.jsonl."""

    chunk_id: str = Field(..., description="Identificador únic del chunk")
    parent_id: str = Field(..., description="Referència al document pare")
    tipus: str = Field(..., description="Tipus d'unitat: Article, Capítol, etc.")
    titol_chunk: str = Field(..., description="Títol o capçalera del chunk")
    text: str = Field(..., description="Contingut textual del chunk")


# ---------------------------------------------------------------------------
# CÀRREGA DE DADES
# ---------------------------------------------------------------------------

def carregar_chunks(path: Path = CHUNKS_INPUT_PATH) -> List[ChildChunk]:
    """
    Carrega i valida els chunks des del fitxer JSON-Lines generat pel chunker.

    Args:
        path: ruta al fitxer chunks.jsonl.

    Returns:
        Llista de ChildChunk validats.

    Raises:
        FileNotFoundError: si el fitxer d'entrada no existeix.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No s'ha trobat {path}. Executa primer src/chunker.py."
        )

    chunks: List[ChildChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for num_linia, linia in enumerate(f, start=1):
            linia = linia.strip()
            if not linia:
                continue
            try:
                dades = json.loads(linia)
                chunks.append(ChildChunk(**dades))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[AVÍS] S'omet la línia {num_linia} per error de format: {e}")

    return chunks


# ---------------------------------------------------------------------------
# TOKENITZACIÓ PER A BM25 (optimitzada per a català)
# ---------------------------------------------------------------------------

# Taula de traducció que elimina la puntuació estàndard més els signes
# tipogràfics habituals en textos catalans escanejats/scrapejats
# (cometes angulars «», cometes altes "", el punt volat "·" de "col·locar").
_TAULA_PUNTUACIO = str.maketrans("", "", string.punctuation + "«»“”·—–")


def tokenitzar(text: str) -> List[str]:
    """
    Tokenitza un text per a l'indexació/cerca BM25.

    Passos:
        1. Minúscules (evita que "Article" i "article" es tractin diferent).
        2. Eliminació de puntuació bàsica i signes tipogràfics catalans.
        3. Col·lapse d'espais múltiples (poden quedar buits en eliminar
           puntuació enganxada entre paraules, p.ex. "això—allò").
        4. Divisió per espais.

    Nota: no s'aplica stemming ni eliminació de stopwords per mantenir la
    funció senzilla i transparent, tal com demana l'especificació; en una
    fase posterior es podria incorporar un stemmer català si cal millorar
    el recall lèxic.

    Args:
        text: text original a tokenitzar.

    Returns:
        Llista de tokens (paraules) en minúscules i sense puntuació.
    """
    text_minuscules = text.lower()
    text_sense_puntuacio = text_minuscules.translate(_TAULA_PUNTUACIO)
    text_normalitzat = re.sub(r"\s+", " ", text_sense_puntuacio).strip()

    if not text_normalitzat:
        return []

    return text_normalitzat.split(" ")


# ---------------------------------------------------------------------------
# INDEXACIÓ SEMÀNTICA (CHROMADB)
# ---------------------------------------------------------------------------

class IndexadorSemantic:
    """
    Encapsula la creació de l'índex vectorial a ChromaDB a partir dels
    chunks Fill, utilitzant el model d'embeddings multilingüe.
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self.collection = self._crear_colleccio_nova()
        self.model = SentenceTransformer(model_name)

    def _crear_colleccio_nova(self) -> "chromadb.Collection":
        """
        Crea la col·lecció 'lleis_andorra' de zero, eliminant-la prèviament
        si ja existia. Això garanteix que cada execució de l'indexador
        reflecteix fidelment l'estat actual de data/chunks.jsonl, sense
        arrossegar chunks obsolets d'execucions anteriors.

        Returns:
            La col·lecció de ChromaDB, buida i llesta per indexar.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # La col·lecció encara no existia: no cal fer res.
            pass
        return self.client.create_collection(name=self.collection_name)

    def indexar(self, chunks: List[ChildChunk], batch_size: int = BATCH_SIZE) -> None:
        """
        Genera els embeddings dels chunks i els insereix a ChromaDB per lots.

        Args:
            chunks: llista de chunks Fill validats.
            batch_size: nombre de chunks a processar per lot.
        """
        total = len(chunks)
        for inici in range(0, total, batch_size):
            lot = chunks[inici : inici + batch_size]

            textos = [c.text for c in lot]
            ids = [c.chunk_id for c in lot]
            # Metadades: parent_id (per recuperar el document pare complet
            # en la fase de generació), tipus i titol_chunk (per filtrar
            # o mostrar resultats sense haver de re-consultar el text).
            metadades = [
                {
                    "parent_id": c.parent_id,
                    "tipus": c.tipus,
                    "titol_chunk": c.titol_chunk,
                }
                for c in lot
            ]

            embeddings = self.model.encode(textos, show_progress_bar=False).tolist()

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=textos,
                metadatas=metadades,
            )

            print(f"[ChromaDB] Indexats {min(inici + batch_size, total)}/{total} chunks")

        print(
            f"[ChromaDB] Indexació semàntica completada. "
            f"Col·lecció '{self.collection.name}' amb {self.collection.count()} elements."
        )


# ---------------------------------------------------------------------------
# INDEXACIÓ LÈXICA (BM25)
# ---------------------------------------------------------------------------

class IndexadorLexic:
    """
    Encapsula la creació de l'índex lèxic BM25 a partir dels chunks Fill.
    """

    def __init__(self, output_path: Path = BM25_INDEX_PATH) -> None:
        self.output_path = output_path

    def indexar(self, chunks: List[ChildChunk]) -> None:
        """
        Tokenitza tots els textos, construeix l'índex BM25Okapi i el
        serialitza a disc amb pickle, juntament amb la llista de
        chunk_id en el mateix ordre que el corpus.

        BM25Okapi no manté cap identificador propi per document: només
        sap treballar amb posicions dins la llista que se li passa. Per
        això és imprescindible guardar `chunk_ids` en paral·lel: la
        posició i-èsima de `chunk_ids` correspon sempre a la posició
        i-èsima del corpus tokenitzat i, per tant, als scores retornats
        per `bm25.get_scores(...)`.

        Args:
            chunks: llista de chunks Fill validats.
        """
        corpus_tokenitzat = [tokenitzar(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus_tokenitzat)

        dades_a_serialitzar = {
            "bm25_index": bm25,
            "chunk_ids": [c.chunk_id for c in chunks],
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("wb") as f:
            pickle.dump(dades_a_serialitzar, f)

        print(
            f"[BM25] Indexació lèxica completada. "
            f"{len(chunks)} documents indexats a {self.output_path}."
        )


# ---------------------------------------------------------------------------
# ORQUESTRACIÓ PRINCIPAL
# ---------------------------------------------------------------------------

def executar_indexacio() -> None:
    """Executa el pipeline complet d'indexació doble (semàntica + lèxica)."""
    chunks = carregar_chunks()

    if not chunks:
        print("[AVÍS] No hi ha chunks per indexar. Revisa data/chunks.jsonl.")
        return

    print(f"S'han carregat {len(chunks)} chunks des de {CHUNKS_INPUT_PATH}\n")

    # --- Indexació semàntica ---
    print("== Indexació semàntica (ChromaDB) ==")
    indexador_semantic = IndexadorSemantic()
    indexador_semantic.indexar(chunks)

    # --- Indexació lèxica ---
    print("\n== Indexació lèxica (BM25) ==")
    indexador_lexic = IndexadorLexic()
    indexador_lexic.indexar(chunks)

    print("\nIndexació doble finalitzada correctament.")


if __name__ == "__main__":
    executar_indexacio()
