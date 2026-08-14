"""
src/retriever.py
=================

Mòdul de la Fase 2 del sistema RAG per a textos legals d'Andorra:
Recuperació Avançada i Re-ranking.

Pipeline de cerca implementat:

    1. Cerca LÈXICA (BM25)      -> top 20 chunk_id per coincidència de termes.
    2. Cerca SEMÀNTICA (Chroma) -> top 20 chunk_id per similitud d'embeddings.
    3. FUSIÓ (Reciprocal Rank Fusion, k=60) -> combina ambdues llistes en
       un únic rànquing de 20 candidats únics, sense necessitat de
       normalitzar scores heterogenis (BM25 i cosinus no són comparables
       directament; RRF només fa servir la posició (rank), no el score brut).
    4. RE-RANKING (Cross-Encoder BAAI/bge-reranker-v2-m3) -> puntua cada
       parella (query, text_candidat) amb un model molt més precís que
       la cerca inicial, i retorna els N millors fragments finals.

Aquest disseny en dues etapes (retrieve ampli + rerank precís) és
l'estàndard de facto en sistemes RAG de producció: la primera etapa
prioritza el recall (no perdre cap document rellevant) i la segona
prioritza la precisió (ordenar bé els pocs candidats finals).

Autor: Enginyeria de Dades RAG
"""

from __future__ import annotations

import json
import pickle
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------------------------------------------------------------------------
# CONFIGURACIÓ GLOBAL
# ---------------------------------------------------------------------------

CHUNKS_PATH = Path("data/chunks.jsonl")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
CHROMA_PERSIST_DIR = Path("data/chroma_db")
COLLECTION_NAME = "lleis_andorra"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Nombre de candidats recuperats a cada branca (lèxica i semàntica) abans
# de fusionar-los. 20 és un valor habitual: prou ampli per no perdre
# recall, prou petit perquè el Cross-Encoder (més costós) sigui ràpid.
TOP_K_PER_BRANCA = 20

# Nombre de candidats únics que es queden després de la fusió RRF i que
# efectivament es passen pel Cross-Encoder.
TOP_K_FUSIO = 20

# Nombre de fragments finals retornats després del re-ranking.
TOP_N_FINAL = 4

# Constant estàndard de l'algorisme Reciprocal Rank Fusion.
RRF_K = 60

# Taula de traducció per a la tokenització BM25 (ha de ser IDÈNTICA a la
# utilitzada a src/indexer.py en indexar, o els tokens no coincidiran).
_TAULA_PUNTUACIO = str.maketrans("", "", string.punctuation + "«»“”·—–")


# ---------------------------------------------------------------------------
# TIPUS DE DADES
# ---------------------------------------------------------------------------

class ResultatFinal(TypedDict):
    """Estructura d'un resultat final retornat al consumidor del retriever."""

    chunk_id: str
    parent_id: str
    tipus: str
    titol_chunk: str
    text: str
    score_cross_encoder: float


# ---------------------------------------------------------------------------
# TOKENITZACIÓ (idèntica a la de l'indexador, per coherència BM25)
# ---------------------------------------------------------------------------

def tokenitzar(text: str) -> List[str]:
    """
    Tokenitza un text per a la cerca BM25: minúscules, sense puntuació,
    dividit per espais.

    IMPORTANT: aquesta funció ha de reproduir exactament la tokenització
    feta durant la indexació (src/indexer.py::tokenitzar). Si divergeixen,
    els scores de BM25 deixen de ser fiables perquè el vocabulari de la
    query no coincidirà amb el vocabulari indexat.

    Args:
        text: text original (query o document) a tokenitzar.

    Returns:
        Llista de tokens en minúscules i sense puntuació.
    """
    text_minuscules = text.lower()
    text_sense_puntuacio = text_minuscules.translate(_TAULA_PUNTUACIO)
    text_normalitzat = re.sub(r"\s+", " ", text_sense_puntuacio).strip()

    if not text_normalitzat:
        return []

    return text_normalitzat.split(" ")


# ---------------------------------------------------------------------------
# CLASSE PRINCIPAL: LegalRetriever
# ---------------------------------------------------------------------------

class LegalRetriever:
    """
    Orquestra la cerca híbrida (BM25 + ChromaDB), la fusió RRF i el
    re-ranking amb Cross-Encoder sobre el corpus de textos legals
    d'Andorra ja indexat (Fase 1).

    Tots els recursos pesants (índexs, models) es carreguen una única
    vegada a `__init__`, de manera que múltiples cerques amb la mateixa
    instància reutilitzen la memòria ja carregada.
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        bm25_index_path: Path = BM25_INDEX_PATH,
        chroma_persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        cross_encoder_model_name: str = CROSS_ENCODER_MODEL_NAME,
    ) -> None:
        print("[LegalRetriever] Carregant recursos...")

        # --- 1. Diccionari chunk_id -> dades del chunk (accés O(1)) ---
        self.chunks_dict: Dict[str, dict] = self._carregar_chunks_dict(chunks_path)
        print(f"  - {len(self.chunks_dict)} chunks carregats a memòria.")

        # --- 2. Índex BM25 + llista de chunk_id en el mateix ordre ---
        self.bm25_index: BM25Okapi
        self.bm25_chunk_ids: List[str]
        self.bm25_index, self.bm25_chunk_ids = self._carregar_bm25(bm25_index_path)
        print(f"  - Índex BM25 carregat ({len(self.bm25_chunk_ids)} documents).")

        # --- 3. Col·lecció ChromaDB persistent ---
        client = chromadb.PersistentClient(path=str(chroma_persist_dir))
        self.collection = client.get_collection(name=collection_name)
        print(f"  - Col·lecció ChromaDB '{collection_name}' carregada "
              f"({self.collection.count()} elements).")

        # --- 4. Model d'embeddings (per vectoritzar la query) ---
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"  - Model d'embeddings '{embedding_model_name}' carregat.")

        # --- 5. Model Cross-Encoder (per al re-ranking final) ---
        self.cross_encoder = CrossEncoder(cross_encoder_model_name)
        print(f"  - Cross-Encoder '{cross_encoder_model_name}' carregat.")

        print("[LegalRetriever] Inicialització completada.\n")

    # -----------------------------------------------------------------
    # CÀRREGA DE RECURSOS
    # -----------------------------------------------------------------

    @staticmethod
    def _carregar_chunks_dict(path: Path) -> Dict[str, dict]:
        """
        Carrega data/chunks.jsonl en un diccionari {chunk_id: dades_chunk}
        per poder recuperar el text d'un chunk en temps O(1) donat el seu
        identificador, sense haver de rellegir el fitxer a cada cerca.

        Args:
            path: ruta al fitxer chunks.jsonl.

        Returns:
            Diccionari chunk_id -> dades completes del chunk.

        Raises:
            FileNotFoundError: si el fitxer no existeix.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"No s'ha trobat {path}. Executa primer src/chunker.py i src/indexer.py."
            )

        chunks_dict: Dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as f:
            for linia in f:
                linia = linia.strip()
                if not linia:
                    continue
                dades = json.loads(linia)
                chunks_dict[dades["chunk_id"]] = dades

        return chunks_dict

    @staticmethod
    def _carregar_bm25(path: Path) -> tuple[BM25Okapi, List[str]]:
        """
        Carrega l'índex BM25 serialitzat i la llista de chunk_id associada.

        Args:
            path: ruta al fitxer bm25_index.pkl.

        Returns:
            Tupla (índex BM25Okapi, llista de chunk_id en el mateix ordre
            que el corpus amb què es va entrenar l'índex).

        Raises:
            FileNotFoundError: si el fitxer no existeix.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"No s'ha trobat {path}. Executa primer src/indexer.py."
            )

        with path.open("rb") as f:
            dades = pickle.load(f)

        return dades["bm25_index"], dades["chunk_ids"]

    # -----------------------------------------------------------------
    # CERCA LÈXICA (BM25)
    # -----------------------------------------------------------------

    def _cercar_bm25(self, query: str, top_k: int = TOP_K_PER_BRANCA) -> List[str]:
        """
        Cerca lèxica amb BM25: tokenitza la query, calcula els scores
        contra tot el corpus indexat i retorna els `top_k` chunk_id amb
        millor puntuació, ordenats de major a menor rellevància.

        Args:
            query: text de la consulta.
            top_k: nombre màxim de resultats a retornar.

        Returns:
            Llista de chunk_id ordenada per rellevància BM25 descendent.
            Els chunks amb score 0 (cap coincidència de termes) s'exclouen,
            ja que no aporten cap senyal lèxic real.
        """
        tokens_query = tokenitzar(query)
        if not tokens_query:
            return []

        scores = self.bm25_index.get_scores(tokens_query)

        # Aparellem cada score amb el seu chunk_id i ordenem descendent.
        resultats = sorted(
            zip(self.bm25_chunk_ids, scores), key=lambda x: x[1], reverse=True
        )

        # Filtrem els resultats amb score 0 (sense cap terme en comú) i
        # ens quedem amb els top_k millors.
        return [chunk_id for chunk_id, score in resultats if score > 0][:top_k]

    # -----------------------------------------------------------------
    # CERCA SEMÀNTICA (CHROMADB)
    # -----------------------------------------------------------------

    def _cercar_semantic(self, query: str, top_k: int = TOP_K_PER_BRANCA) -> List[str]:
        """
        Cerca semàntica: genera l'embedding de la query i consulta
        ChromaDB per similitud vectorial, retornant els `top_k` chunk_id
        més propers semànticament.

        Args:
            query: text de la consulta.
            top_k: nombre màxim de resultats a retornar.

        Returns:
            Llista de chunk_id ordenada per similitud semàntica descendent
            (ChromaDB ja retorna els resultats ordenats per distància).
        """
        embedding_query = self.embedding_model.encode(query).tolist()

        resultats = self.collection.query(
            query_embeddings=[embedding_query],
            n_results=top_k,
        )

        # Els IDs dels documents a ChromaDB SÓN els chunk_id (es van
        # inserir així a l'indexador), de manera que no cal passar per
        # les metadades: n'hi ha prou amb `ids`.
        ids_resultat = resultats.get("ids", [[]])[0]
        return list(ids_resultat)

    # -----------------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # -----------------------------------------------------------------

    @staticmethod
    def _fusionar_rrf(
        llista_bm25: List[str],
        llista_semantica: List[str],
        k: int = RRF_K,
        top_k: int = TOP_K_FUSIO,
    ) -> List[str]:
        """
        Combina dues llistes de chunk_id (una per branca de cerca) en un
        únic rànquing mitjançant Reciprocal Rank Fusion (RRF).

        Per a cada document `d`:
            RRF_score(d) = 1/(k + rank_BM25(d)) + 1/(k + rank_Chroma(d))

        On rank_X(d) és la posició de `d` dins la llista X (començant a 1).
        Si `d` no apareix en una de les llistes, aquell sumand es considera
        0 (no penalitza, simplement no suma contribució d'aquella branca).

        RRF té l'avantatge de no necessitar normalitzar els scores bruts
        de BM25 (no acotats) i de ChromaDB (distàncies/similituds), ja que
        només es basa en la POSICIÓ relativa dins de cada llista.

        Args:
            llista_bm25: chunk_id ordenats per rellevància BM25.
            llista_semantica: chunk_id ordenats per rellevància semàntica.
            k: constant de suavitzat de RRF (per defecte 60, valor estàndard).
            top_k: nombre de candidats únics a retornar després de fusionar.

        Returns:
            Llista dels `top_k` chunk_id únics amb millor RRF_score,
            ordenada de major a menor score.
        """
        # Construïm {chunk_id: rank} (rank 1-indexat) per a cada branca.
        rangs_bm25 = {chunk_id: rank for rank, chunk_id in enumerate(llista_bm25, start=1)}
        rangs_semantica = {
            chunk_id: rank for rank, chunk_id in enumerate(llista_semantica, start=1)
        }

        # Unió de tots els chunk_id candidats (presents en almenys una branca).
        tots_els_ids = set(rangs_bm25) | set(rangs_semantica)

        scores_rrf: Dict[str, float] = {}
        for chunk_id in tots_els_ids:
            score = 0.0
            if chunk_id in rangs_bm25:
                score += 1.0 / (k + rangs_bm25[chunk_id])
            if chunk_id in rangs_semantica:
                score += 1.0 / (k + rangs_semantica[chunk_id])
            scores_rrf[chunk_id] = score

        # Ordenem per score RRF descendent i ens quedem amb els top_k.
        ids_ordenats = sorted(scores_rrf, key=lambda cid: scores_rrf[cid], reverse=True)
        return ids_ordenats[:top_k]

    # -----------------------------------------------------------------
    # RE-RANKING AMB CROSS-ENCODER
    # -----------------------------------------------------------------

    def _rerank_cross_encoder(
        self, query: str, chunk_ids_candidats: List[str], top_n: int = TOP_N_FINAL
    ) -> List[ResultatFinal]:
        """
        Refina l'ordenació dels candidats fusionats fent servir un model
        Cross-Encoder, molt més precís que BM25/embeddings de bi-encoder
        perquè processa conjuntament (query, document) en comptes de
        comparar representacions vectorials independents.

        Args:
            query: text de la consulta original.
            chunk_ids_candidats: chunk_id sortits de la fusió RRF.
            top_n: nombre de fragments finals a retornar.

        Returns:
            Llista dels `top_n` ResultatFinal ordenats per score del
            Cross-Encoder descendent.
        """
        # Recuperem el text i les metadades de cada candidat des del
        # diccionari en memòria. Si algun chunk_id de l'índex ja no
        # existeix a chunks.jsonl (p.ex. per una reindexació parcial),
        # se salta silenciosament en comptes de trencar la cerca.
        candidats_valids = [
            self.chunks_dict[cid] for cid in chunk_ids_candidats if cid in self.chunks_dict
        ]

        if not candidats_valids:
            return []

        parelles = [(query, candidat["text"]) for candidat in candidats_valids]
        scores = self.cross_encoder.predict(parelles)

        candidats_amb_score = list(zip(candidats_valids, scores))
        candidats_amb_score.sort(key=lambda x: x[1], reverse=True)

        resultats_finals: List[ResultatFinal] = [
            ResultatFinal(
                chunk_id=candidat["chunk_id"],
                parent_id=candidat["parent_id"],
                tipus=candidat["tipus"],
                titol_chunk=candidat["titol_chunk"],
                text=candidat["text"],
                score_cross_encoder=float(score),
            )
            for candidat, score in candidats_amb_score[:top_n]
        ]

        return resultats_finals

    # -----------------------------------------------------------------
    # MÈTODE PÚBLIC PRINCIPAL: PIPELINE COMPLET DE CERCA
    # -----------------------------------------------------------------

    def cercar(
        self,
        query: str,
        top_k_per_branca: int = TOP_K_PER_BRANCA,
        top_k_fusio: int = TOP_K_FUSIO,
        top_n_final: int = TOP_N_FINAL,
    ) -> List[ResultatFinal]:
        """
        Executa el pipeline complet de recuperació avançada:

            1. Cerca lèxica (BM25)      -> top_k_per_branca candidats.
            2. Cerca semàntica (Chroma) -> top_k_per_branca candidats.
            3. Fusió RRF                -> top_k_fusio candidats únics.
            4. Re-ranking Cross-Encoder -> top_n_final fragments finals.

        Args:
            query: pregunta o consulta en llenguatge natural.
            top_k_per_branca: nombre de resultats a recuperar de cada
                branca (BM25 i ChromaDB) abans de fusionar.
            top_k_fusio: nombre de candidats únics conservats després
                de la fusió RRF (i que es passen pel Cross-Encoder).
            top_n_final: nombre de fragments finals retornats.

        Returns:
            Llista dels `top_n_final` fragments més rellevants, ordenats
            per score del Cross-Encoder descendent.
        """
        llista_bm25 = self._cercar_bm25(query, top_k=top_k_per_branca)
        llista_semantica = self._cercar_semantic(query, top_k=top_k_per_branca)

        candidats_fusionats = self._fusionar_rrf(
            llista_bm25, llista_semantica, k=RRF_K, top_k=top_k_fusio
        )

        return self._rerank_cross_encoder(query, candidats_fusionats, top_n=top_n_final)


# ---------------------------------------------------------------------------
# PUNT D'ENTRADA: EXEMPLE D'ÚS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = LegalRetriever()

    query_exemple = "Quins són els deures dels voluntaris?"
    print(f"Cerca: \"{query_exemple}\"\n")

    resultats = retriever.cercar(query_exemple)

    if not resultats:
        print("No s'ha trobat cap resultat rellevant.")
    else:
        for i, resultat in enumerate(resultats, start=1):
            print(f"--- Resultat {i} (score={resultat['score_cross_encoder']:.4f}) ---")
            print(f"chunk_id:    {resultat['chunk_id']}")
            print(f"parent_id:   {resultat['parent_id']}")
            print(f"tipus:       {resultat['tipus']}")
            print(f"títol:       {resultat['titol_chunk']}")
            print(f"text:        {resultat['text'][:300]}...")
            print()
