"""
src/chunker.py
===============

Mòdul 1 de la Fase 1 del sistema RAG per a textos legals d'Andorra.

Llegeix els fitxers de text ja net a `data/clean_texts/` (un .txt per llei),
organitzats en subcarpetes per any (p.ex. `data/clean_texts/2010/Llei_94_2010.txt`),
i els descompon en una estructura jeràrquica de dos nivells:

    - Document PARE  -> la llei sencera (nom d'arxiu / títol + text complet)
    - Chunks FILL    -> articles, capítols o disposicions, cadascun
                         vinculat al seu pare mitjançant `parent_id`.

Sortides:
    - data/parent_store.json   -> {parent_id: {...dades del pare...}}
    - data/chunks.jsonl        -> un ChildChunk (JSON) per línia

Autor: Enginyeria de Dades RAG
"""

from __future__ import annotations

import json
import re
import unicodedata
import warnings
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# CONFIGURACIÓ GLOBAL
# ---------------------------------------------------------------------------

CLEAN_TEXTS_DIR = Path("data/clean_texts")
PARENT_STORE_PATH = Path("data/parent_store.json")
CHUNKS_OUTPUT_PATH = Path("data/chunks.jsonl")

# Patró per detectar si el nom d'una carpeta és un any de 4 xifres
# (p.ex. "2010"), utilitzat per enriquir les metadades del document Pare
# quan els fitxers estan organitzats en subcarpetes per any.
ANY_CARPETA_PATTERN = re.compile(r"^(19|20)\d{2}$")


# ---------------------------------------------------------------------------
# REGEX D'ESTRUCTURA LEGAL
# ---------------------------------------------------------------------------
#
# Explicació del disseny del patró:
#
#   ^(?P<tipus_base>Article|Capítol|Disposici[óons]+)
#       -> Captura la paraula clau que obre la unitat estructural.
#          "Disposici[óons]+" cobreix singular ("Disposició") i plural
#          ("Disposicions"), ja que als reglaments és habitual trobar
#          capçaleres plurals com "Disposicions transitòries".
#
#   (?:\s+(?P<subtipus>addicional|transitòri[a-zà-ú]*|final(?:s)?|
#          derogatòri[a-zà-ú]*))?
#       -> Captura opcionalment el subtipus de disposició (addicional,
#          transitòria, final, derogatòria), en singular o plural.
#
#   \s*(?P<num>[0-9]+(?:\s+(?:bis|ter|quater))?|[IVXLCDM]+|únic(?:a)?)?
#       -> Captura el número identificador de l'article/capítol:
#            - numeració aràbiga, amb possible sufix "bis"/"ter"/"quater"
#              (p.ex. "Article 28 bis.")
#            - numeració romana (p.ex. "Capítol IV")
#            - la paraula "únic"/"única" (p.ex. "Article únic")
#
#   \.?\s*(?P<resta_titol>.*)$
#       -> La resta de la línia (si n'hi ha) es tracta com a títol
#          descriptiu del chunk (p.ex. "Naturalesa i àmbit d'aplicació").
#
# El patró es compila amb re.MULTILINE perquè ha de coincidir a l'inici
# de cada línia del document (no només a l'inici del text sencer).
# ---------------------------------------------------------------------------

STRUCTURE_PATTERN = re.compile(
    r"^(?P<tipus_base>Article|Capítol|Disposici[óons]+)"
    r"(?:\s+(?P<subtipus>addicional(?:s)?|transitòri[a-zà-ú]*|final(?:s)?|derogatòri[a-zà-ú]*))?"
    r"\s*(?P<num>[0-9]+(?:\s+(?:bis|ter|quater))?|[IVXLCDM]+|únic(?:a)?)?"
    r"\.?\s*(?P<resta_titol>.*)$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# MODELS DE DADES (Pydantic) — validen l'estructura abans de persistir
# ---------------------------------------------------------------------------

class ParentDocument(BaseModel):
    """Representa el document 'Pare': la llei/reglament sencer."""

    parent_id: str = Field(..., description="Identificador únic del document")
    titol: str = Field(..., description="Títol de la llei (primera línia o nom de fitxer)")
    font_fitxer: str = Field(..., description="Nom del fitxer d'origen")
    any_publicacio: Optional[str] = Field(
        default=None,
        description="Any de publicació, inferit del nom de la subcarpeta (p.ex. '2010')",
    )
    text_complet: str = Field(..., description="Text net sencer del document")

    @field_validator("text_complet")
    @classmethod
    def text_no_buit(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("El text complet del document pare no pot ser buit.")
        return v


class ChildChunk(BaseModel):
    """Representa un chunk 'Fill': un article, capítol o disposició."""

    chunk_id: str = Field(..., description="Identificador únic i llegible del chunk")
    parent_id: str = Field(..., description="Referència al document pare")
    tipus: str = Field(..., description="Tipus d'unitat: Article, Capítol, Disposició, etc.")
    titol_chunk: str = Field(..., description="Títol o capçalera del chunk")
    text: str = Field(..., description="Contingut sencer del chunk fins al següent tall")

    @field_validator("text")
    @classmethod
    def text_no_buit(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("El text d'un chunk fill no pot ser buit.")
        return v


# ---------------------------------------------------------------------------
# UTILITATS DE NORMALITZACIÓ DE TEXT
# ---------------------------------------------------------------------------

def _normalitzar_text(text: str) -> str:
    """
    Normalitza el text llegit del fitxer:
      - Converteix finals de línia Windows (\\r\\n) a Unix (\\n).
      - Elimina espais sobrants a cada línia.
      - Elimina línies completament buides (col·lapsa separadors múltiples).

    Args:
        text: text original llegit del disc.

    Returns:
        Text normalitzat, una unitat lògica per línia.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    linies = [linia.strip() for linia in text.split("\n")]
    linies_netes = [linia for linia in linies if linia]
    return "\n".join(linies_netes)


def _slug(text: str) -> str:
    """
    Converteix un fragment de text en un 'slug' apte per a chunk_id:
    minúscules, sense accents ni caràcters especials, espais -> guions baixos.

    Args:
        text: text a normalitzar (p.ex. "28 bis" -> "28_bis").

    Returns:
        Cadena en minúscules, ascii, separada per '_'.
    """
    # Descomponem els caràcters Unicode (NFKD) i eliminem els diacrítics.
    text_sense_accents = unicodedata.normalize("NFKD", text)
    text_sense_accents = "".join(
        c for c in text_sense_accents if not unicodedata.combining(c)
    )
    text_net = re.sub(r"[^a-zA-Z0-9]+", "_", text_sense_accents).strip("_")
    return text_net.lower()


# ---------------------------------------------------------------------------
# LECTURA DE FITXERS AMB GESTIÓ D'ERRORS
# ---------------------------------------------------------------------------

def llegir_fitxer_segur(path_fitxer: Path) -> Optional[str]:
    """
    Llegeix un fitxer .txt de manera segura, capturant errors de lectura
    o codificació. Si el fitxer és il·legible o resulta buit un cop
    normalitzat, es registra un avís i es retorna None perquè el
    document sigui omès pel pipeline.

    Args:
        path_fitxer: ruta al fitxer d'entrada.

    Returns:
        Text normalitzat, o None si el fitxer no s'ha pogut processar.
    """
    try:
        contingut_brut = path_fitxer.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        # Alguns fitxers de scraping antic poden venir en Latin-1 (ISO-8859-1).
        try:
            contingut_brut = path_fitxer.read_text(encoding="latin-1")
            warnings.warn(
                f"{path_fitxer.name}: llegit amb codificació de reserva 'latin-1'."
            )
        except Exception as e:
            warnings.warn(f"{path_fitxer.name}: fitxer il·legible ({e}). S'omet.")
            return None
    except OSError as e:
        warnings.warn(f"{path_fitxer.name}: error d'accés al fitxer ({e}). S'omet.")
        return None

    text_net = _normalitzar_text(contingut_brut)

    if not text_net:
        warnings.warn(f"{path_fitxer.name}: fitxer buit després de normalitzar. S'omet.")
        return None

    return text_net


# ---------------------------------------------------------------------------
# EXTRACCIÓ DEL TÍTOL DEL DOCUMENT PARE
# ---------------------------------------------------------------------------

def extreure_any_de_ruta(path_fitxer: Path, arrel: Path) -> Optional[str]:
    """
    Intenta inferir l'any de publicació a partir del nom de la subcarpeta
    immediata que conté el fitxer (p.ex. data/clean_texts/2010/x.txt -> "2010").

    Si el fitxer és directament a l'arrel de `arrel` (sense subcarpeta
    d'any), o si el nom de la carpeta no sembla un any de 4 xifres, es
    retorna None sense provocar cap error.

    Args:
        path_fitxer: ruta completa al fitxer.
        arrel: carpeta arrel de cerca (CLEAN_TEXTS_DIR), per no confondre
            l'arrel mateixa amb una carpeta d'any.

    Returns:
        L'any com a cadena de 4 xifres, o None si no s'ha pogut inferir.
    """
    carpeta_pare = path_fitxer.parent
    if carpeta_pare == arrel:
        return None

    if ANY_CARPETA_PATTERN.match(carpeta_pare.name):
        return carpeta_pare.name

    return None


def extreure_titol(text_net: str, nom_fitxer: str) -> str:
    """
    Extreu el títol de la llei/reglament fent servir la primera línia no
    buida del document net. Si per algun motiu no hi ha cap línia
    disponible, es fa servir el nom del fitxer com a reserva (fallback).

    Args:
        text_net: text ja normalitzat del document.
        nom_fitxer: nom del fitxer d'origen (fallback).

    Returns:
        Títol de la llei.
    """
    primera_linia = text_net.split("\n", 1)[0].strip()
    if primera_linia:
        return primera_linia
    return Path(nom_fitxer).stem.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# SEGMENTACIÓ EN CHUNKS FILL (VIA REGEX)
# ---------------------------------------------------------------------------

def _construir_tipus_llegible(tipus_base: str, subtipus: Optional[str]) -> str:
    """
    Construeix l'etiqueta 'tipus' llegible del chunk a partir de la
    paraula clau base i el subtipus opcional.

    Exemples:
        ("Article", None)                 -> "Article"
        ("Capítol", None)                 -> "Capítol"
        ("Disposició", "addicional")      -> "Disposició addicional"
        ("Disposicions", "transitòries")  -> "Disposicions transitòries"

    Args:
        tipus_base: paraula clau capturada (Article/Capítol/Disposici...).
        subtipus: qualificador opcional (addicional/transitòria/final/...).

    Returns:
        Etiqueta de tipus normalitzada i capitalitzada.
    """
    tipus_base_cap = tipus_base.strip().capitalize()
    if subtipus:
        return f"{tipus_base_cap} {subtipus.strip().lower()}"
    return tipus_base_cap


def _construir_chunk_id(prefix_chunk_id: str, tipus_base: str, num: Optional[str], comptador: int) -> str:
    """
    Construeix un chunk_id llegible i únic, amb el format:
        <prefix>_<tipus>_<num>
    p.ex. "2010_llei_94_2010_article_28_bis"

    Si el chunk no té número identificable (p.ex. una capçalera de
    "Disposicions transitòries" sense numeral), es fa servir un comptador
    seqüencial per garantir la unicitat.

    Args:
        prefix_chunk_id: prefix ja en format slug (any + nom de fitxer,
            o només nom de fitxer si l'any no es coneix).
        tipus_base: paraula clau base (Article/Capítol/Disposici...).
        num: número/identificador capturat pel regex (pot ser None).
        comptador: índex seqüencial de reserva per si no hi ha número.

    Returns:
        chunk_id en format slug, únic dins del corpus.
    """
    tipus_slug = _slug(tipus_base)
    identificador = _slug(num) if num else str(comptador)
    return f"{prefix_chunk_id}_{tipus_slug}_{identificador}"


def segmentar_en_chunks(text_net: str, parent_id: str, prefix_chunk_id: str) -> List[ChildChunk]:
    """
    Divideix el text net d'un document legal en chunks 'Fill' (articles,
    capítols, disposicions) mitjançant STRUCTURE_PATTERN.

    Estratègia: es localitzen totes les línies on comença una nova unitat
    estructural; el text de cada chunk és el fragment comprès entre
    l'inici d'una unitat i l'inici de la següent (o el final del document),
    de manera que cada Fill conté el seu contingut sencer.

    Args:
        text_net: text net i complet del document.
        parent_id: identificador del document pare al qual pertanyen els fills.
        prefix_chunk_id: prefix (ja en format slug) utilitzat per construir
            chunk_id únics, típicament "<any>_<nom_fitxer>" o "<nom_fitxer>"
            quan no es coneix l'any. Es passa ja calculat des del pipeline
            perquè coincideixi exactament amb el prefix usat al parent_id.

    Returns:
        Llista de ChildChunk validats amb Pydantic.
    """
    coincidencies = list(STRUCTURE_PATTERN.finditer(text_net))
    chunks: List[ChildChunk] = []

    if not coincidencies:
        # Si no es detecta cap estructura reconeguda, es conserva tot el
        # text com a únic chunk, per no perdre contingut de la llei.
        chunks.append(
            ChildChunk(
                chunk_id=f"{prefix_chunk_id}_document_complet",
                parent_id=parent_id,
                tipus="Document",
                titol_chunk="Text complet (sense estructura detectada)",
                text=text_net,
            )
        )
        return chunks

    comptador_sense_num = 0
    # Registre de chunk_id ja assignats dins d'aquest document. És una
    # xarxa de seguretat addicional al comptador_sense_num: alguns textos
    # legals repeteixen literalment la mateixa capçalera (p.ex. "Capítol
    # únic" apareix tant a l'índex/sumari inicial com al cos del document,
    # o una numeració es duplica per error d'origen). Com que en aquests
    # casos `num` no és None, el comptador_sense_num no s'activa, i sense
    # aquesta deduplicació ChromaDB rebutjaria la càrrega per IDs repetits.
    ids_utilitzats: dict[str, int] = {}

    for idx, coincidencia in enumerate(coincidencies):
        inici = coincidencia.start()
        final = (
            coincidencies[idx + 1].start()
            if idx + 1 < len(coincidencies)
            else len(text_net)
        )

        fragment = text_net[inici:final].strip()
        if not fragment:
            continue

        tipus_base = coincidencia.group("tipus_base")
        subtipus = coincidencia.group("subtipus")
        num = coincidencia.group("num")
        resta_titol = (coincidencia.group("resta_titol") or "").strip()

        tipus_llegible = _construir_tipus_llegible(tipus_base, subtipus)

        if not num:
            comptador_sense_num += 1
        chunk_id_base = _construir_chunk_id(prefix_chunk_id, tipus_base, num, comptador_sense_num)

        # Deduplicació: si aquest chunk_id ja s'ha fet servir en aquest
        # mateix document, hi afegim un sufix numèric incremental
        # (_2, _3, ...) fins a garantir unicitat.
        if chunk_id_base in ids_utilitzats:
            ids_utilitzats[chunk_id_base] += 1
            chunk_id = f"{chunk_id_base}_dup{ids_utilitzats[chunk_id_base]}"
            warnings.warn(
                f"chunk_id duplicat detectat i corregit: "
                f"'{chunk_id_base}' -> '{chunk_id}' (parent_id={parent_id})"
            )
        else:
            ids_utilitzats[chunk_id_base] = 1
            chunk_id = chunk_id_base

        # Construïm un títol llegible, p.ex. "Article 1. Objecte"
        capcalera = f"{tipus_llegible} {num}".strip() if num else tipus_llegible
        titol_chunk = f"{capcalera}. {resta_titol}" if resta_titol else capcalera

        try:
            chunks.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    tipus=tipus_llegible,
                    titol_chunk=titol_chunk,
                    text=fragment,
                )
            )
        except ValueError as e:
            warnings.warn(f"Chunk omès per error de validació ({chunk_id}): {e}")

    return chunks


# ---------------------------------------------------------------------------
# CLASSE PRINCIPAL: ORQUESTRADORA DEL PROCÉS DE CHUNKING
# ---------------------------------------------------------------------------

class ChunkerPipeline:
    """
    Orquestra tot el procés de chunking: llegeix els fitxers de
    data/clean_texts/, en genera els documents Pare i els chunks Fill,
    i els persisteix a disc en els formats esperats pel mòdul d'indexació.
    """

    def __init__(
        self,
        clean_texts_dir: Path = CLEAN_TEXTS_DIR,
        parent_store_path: Path = PARENT_STORE_PATH,
        chunks_output_path: Path = CHUNKS_OUTPUT_PATH,
    ) -> None:
        self.clean_texts_dir = clean_texts_dir
        self.parent_store_path = parent_store_path
        self.chunks_output_path = chunks_output_path

    def _llistar_fitxers_entrada(self) -> List[Path]:
        """
        Retorna tots els fitxers .txt de la carpeta d'entrada, ordenats.

        Es fa servir `rglob` (cerca recursiva) en lloc de `glob` perquè els
        fitxers estan organitzats en subcarpetes per any
        (p.ex. data/clean_texts/2010/Llei_94_2010.txt), no directament
        a l'arrel de data/clean_texts/.
        """
        if not self.clean_texts_dir.exists():
            warnings.warn(f"La carpeta {self.clean_texts_dir} no existeix.")
            return []
        return sorted(self.clean_texts_dir.rglob("*.txt"))

    def processar_fitxer(self, path_fitxer: Path) -> Optional[tuple[ParentDocument, List[ChildChunk]]]:
        """
        Processa un únic fitxer: el llegeix de manera segura, en construeix
        el document Pare i en segmenta el contingut en chunks Fill.

        Args:
            path_fitxer: ruta al fitxer d'entrada.

        Returns:
            Tupla (ParentDocument, llista de ChildChunk), o None si el
            fitxer ha estat omès per un error de lectura o validació.
        """
        text_net = llegir_fitxer_segur(path_fitxer)
        if text_net is None:
            return None

        stem_fitxer = path_fitxer.stem
        titol = extreure_titol(text_net, path_fitxer.name)
        any_publicacio = extreure_any_de_ruta(path_fitxer, self.clean_texts_dir)

        # Prefix comú per a parent_id i chunk_id: incloure l'any (quan es
        # coneix) evita col·lisions entre fitxers amb el mateix nom ubicats
        # en carpetes d'anys diferents.
        if any_publicacio:
            prefix_chunk_id = f"{any_publicacio}_{_slug(stem_fitxer)}"
        else:
            prefix_chunk_id = _slug(stem_fitxer)
        parent_id = f"parent_{prefix_chunk_id}"

        try:
            document_pare = ParentDocument(
                parent_id=parent_id,
                titol=titol,
                font_fitxer=path_fitxer.name,
                any_publicacio=any_publicacio,
                text_complet=text_net,
            )
        except ValueError as e:
            warnings.warn(f"{path_fitxer.name}: document pare invàlid ({e}). S'omet.")
            return None

        chunks_fill = segmentar_en_chunks(text_net, parent_id, prefix_chunk_id)

        return document_pare, chunks_fill

    def executar(self) -> None:
        """Executa el pipeline complet sobre tots els fitxers i els persisteix."""
        fitxers = self._llistar_fitxers_entrada()
        if not fitxers:
            print(f"[AVÍS] No s'ha trobat cap fitxer .txt a {self.clean_texts_dir}")
            return

        parent_store: dict[str, dict] = {}
        tots_els_chunks: List[ChildChunk] = []

        for fitxer in fitxers:
            try:
                resultat = self.processar_fitxer(fitxer)
            except Exception as e:
                # Xarxa de seguretat final: cap fitxer individual ha de
                # poder aturar tot el pipeline.
                warnings.warn(f"{fitxer.name}: error inesperat ({e}). S'omet.")
                continue

            if resultat is None:
                continue

            document_pare, chunks_fill = resultat
            parent_store[document_pare.parent_id] = document_pare.model_dump()
            tots_els_chunks.extend(chunks_fill)

            print(
                f"[OK] {fitxer.name} -> {len(chunks_fill)} chunks "
                f"(parent_id={document_pare.parent_id})"
            )

        self._guardar_parent_store(parent_store)
        self._guardar_chunks(tots_els_chunks)

        print(
            f"\nFinalitzat: {len(parent_store)} documents pare, "
            f"{len(tots_els_chunks)} chunks fill generats."
        )

    def _guardar_parent_store(self, parent_store: dict[str, dict]) -> None:
        """Persisteix el diccionari {parent_id: dades} en format JSON."""
        self.parent_store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.parent_store_path.open("w", encoding="utf-8") as f:
            json.dump(parent_store, f, ensure_ascii=False, indent=2)

    def _guardar_chunks(self, chunks: List[ChildChunk]) -> None:
        """Persisteix la llista de chunks Fill en format JSON-Lines."""
        self.chunks_output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.chunks_output_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# PUNT D'ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Assegurem que els avisos (warnings) es mostrin sempre per pantalla,
    # ja que és el mecanisme que utilitzem per registrar fitxers omesos.
    warnings.simplefilter("always")
    pipeline = ChunkerPipeline()
    pipeline.executar()