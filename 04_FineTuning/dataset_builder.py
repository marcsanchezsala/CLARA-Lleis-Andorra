"""
CLARA — Corpus Legal Andorrà amb Recuperació Augmentada
Fase 3: Generació automàtica del dataset d'entrenament

Genera parells instrucció-resposta a partir dels chunks legals,
usant Ollama en local com a model generador.

Ús:
    python fase3_dataset_builder.py

Prerequisits:
    ollama serve          (en una altra terminal)
    ollama pull llama3    (o qualsevol model instal·lat)

Entrada:
    data/chunks.jsonl     → generat per fase1_chunker.py

Sortida:
    data/train_data.jsonl → dataset d'entrenament per a fase3_finetune_colab.ipynb
"""

import json
import logging
import random
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Configuració ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CHUNKS_FILE   = Path("data/chunks.jsonl")
OUTPUT_FILE   = Path("data/train_data.jsonl")

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.2"     # Canvia si tens un altre model (ex: mistral, qwen2.5)

TARGET_EXAMPLES = 500          # Nombre d'exemples a generar
MIN_CHUNK_CHARS = 200          # Chunks massa curts no generen bones preguntes
MAX_RETRIES     = 3            # Reintents per chunk si falla la generació
SAVE_EVERY      = 25           # Desar progrés cada N exemples


# ── Prompt de generació ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ets un expert jurídic especialitzat en la legislació del Principat d'Andorra.
La teva tasca és generar exemples d'entrenament per a un model d'IA jurídic.

Donat un fragment de text legal andorrà, has de generar:
1. Una pregunta natural que un ciutadà o professional podria fer sobre el contingut
2. Una resposta completa, precisa i en català correcte, citant l'article i llei corresponents

Respon ÚNICAMENT amb un objecte JSON vàlid, sense cap text addicional, en aquest format exacte:
{
  "instruction": "la pregunta aquí",
  "input": "",
  "output": "la resposta completa aquí"
}"""

USER_TEMPLATE = """Fragment legal:
---
{chunk_text}
---
Font: {source_title}

Genera un parell pregunta-resposta basat en aquest fragment."""


# ── Funcions ──────────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def filter_chunks(chunks: list[dict]) -> list[dict]:
    """Selecciona chunks aptes per generar preguntes de qualitat."""
    good = [
        c for c in chunks
        if c["n_chars"] >= MIN_CHUNK_CHARS
        and "Article" in c["text"]  # Preferir chunks amb articles numerats
    ]
    logging.info(f"Chunks aptes (≥{MIN_CHUNK_CHARS} chars + Article): {len(good)} / {len(chunks)}")
    return good


def call_ollama(chunk: dict) -> dict | None:
    """Crida Ollama i retorna el parell instrucció-resposta parseajat."""
    prompt = USER_TEMPLATE.format(
        chunk_text=chunk["text"],
        source_title=chunk["source_title"],
    )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        raw_text = response.json().get("response", "")

        # Extreure el JSON de la resposta (pot tenir text al voltant)
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            return None

        parsed = json.loads(json_match.group())

        # Validar camps obligatoris
        if not parsed.get("instruction") or not parsed.get("output"):
            return None

        # Afegir metadades de traçabilitat
        parsed["source_chunk_id"] = chunk["chunk_id"]
        parsed["source_title"]    = chunk["source_title"]
        parsed["source_url"]      = chunk["source_url"]

        return parsed

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        logging.debug(f"Error a call_ollama: {e}")
        return None


def check_ollama() -> bool:
    """Comprova que Ollama està en marxa i el model existeix."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        available = any(OLLAMA_MODEL in m for m in models)
        if not available:
            logging.error(
                f"Model '{OLLAMA_MODEL}' no trobat. Models disponibles: {models}\n"
                f"Executa: ollama pull {OLLAMA_MODEL}"
            )
        else:
            logging.info(f"Ollama OK — model '{OLLAMA_MODEL}' disponible.")
        return available
    except requests.RequestException:
        logging.error(
            "Ollama no respon a http://localhost:11434\n"
            "Executa: ollama serve"
        )
        return False


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    """Carrega exemples ja generats per reprendre una execució interrompuda."""
    if not path.exists():
        return [], set()
    examples = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
            seen_ids.add(ex.get("source_chunk_id", ""))
    logging.info(f"Exemples ja generats (reprenen execució): {len(examples)}")
    return examples, seen_ids


# ── Pipeline principal ────────────────────────────────────────────────────────

def run():
    if not CHUNKS_FILE.exists():
        logging.error(f"No s'ha trobat {CHUNKS_FILE}. Executa primer fase1_chunker.py")
        return

    if not check_ollama():
        return

    chunks    = load_chunks(CHUNKS_FILE)
    good      = filter_chunks(chunks)

    # Reprendre si hi ha execució prèvia
    examples, seen_ids = load_existing(OUTPUT_FILE)
    remaining_target   = TARGET_EXAMPLES - len(examples)

    if remaining_target <= 0:
        logging.info(f"Ja tens {len(examples)} exemples. TARGET_EXAMPLES assolit.")
        return

    # Barrejar i filtrar chunks ja processats
    random.seed(42)
    random.shuffle(good)
    pending = [c for c in good if c["chunk_id"] not in seen_ids]

    logging.info(f"Chunks pendents: {len(pending)} | Objectiu restant: {remaining_target}")

    # Obrir fitxer en mode append
    out_file = OUTPUT_FILE.open("a", encoding="utf-8")
    errors   = 0

    with tqdm(total=remaining_target, desc="Generant exemples", unit="ex") as pbar:
        for chunk in pending:
            if len(examples) - (TARGET_EXAMPLES - remaining_target) >= remaining_target:
                break

            success = False
            for attempt in range(MAX_RETRIES):
                result = call_ollama(chunk)
                if result:
                    examples.append(result)
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    pbar.update(1)
                    pbar.set_postfix({"errors": errors, "total": len(examples)})
                    success = True
                    break
                time.sleep(1)  # Petita pausa entre reintents

            if not success:
                errors += 1
                tqdm.write(f"  ⚠ Error generant exemple per: {chunk['source_title'][:60]}")

            # Flush periòdic
            out_file.flush()

    out_file.close()

    # Estadístiques finals
    logging.info(f"\n{'='*50}")
    logging.info(f"Exemples generats:  {len(examples)}")
    logging.info(f"Errors / omesos:    {errors}")
    logging.info(f"Dataset desat a:    {OUTPUT_FILE}")
    logging.info(f"\nProper pas: puja {OUTPUT_FILE} a Google Colab i executa fase3_finetune_colab.ipynb")


if __name__ == "__main__":
    run()
