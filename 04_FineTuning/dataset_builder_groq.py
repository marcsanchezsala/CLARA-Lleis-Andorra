"""
CLARA — Corpus Legal Andorrà amb Recuperació Augmentada
Fase 3: Generació automàtica del dataset d'entrenament (versió Groq/Colab)

Substitueix Ollama per l'API gratuïta de Groq (Llama 3 70B).
Dissenyat per executar-se a Google Colab.

═══════════════════════════════════════════════════════
  INSTRUCCIONS DE CONFIGURACIÓ (llegir abans d'executar)
═══════════════════════════════════════════════════════

1. Obtenir API key gratuïta de Groq:
      https://console.groq.com  →  API Keys  →  Create API Key

2. Afegir la key a Colab Secrets:
      Colab: icona 🔑 (esquerra)  →  "+ Add new secret"
      Nom:   GROQ_API_KEY
      Valor: gsk_xxxxxxxxxxxxxxxxxxxx

3. Pujar chunks_top1000.jsonl a Colab:
      Colab: icona 📁 (esquerra)  →  arrossegar el fitxer

4. Executar totes les cel·les en ordre.
      El script muntarà Google Drive automàticament i desarà
      una còpia de seguretat cada 25 exemples a:
      MyDrive/CLARA/train_data.jsonl

═══════════════════════════════════════════════════════

Entrada:  chunks_top1000.jsonl   (o chunks.jsonl complet)
Sortida:  train_data.jsonl       → per a fase3_finetune_colab.ipynb

Límits Groq free tier:
    - 14.400 req/dia  amb llama-3.3-70b-versatile
    - ~500 tokens/s   (molt ràpid)
    → 500 exemples ≈ 10-15 minuts
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CEL·LA 1 — Instal·lar dependències
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Executa això a Colab:
#   !pip install groq tqdm -q

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CEL·LA 2 — Codi principal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import logging
import random
import re
import time
from pathlib import Path

from groq import Groq
from tqdm import tqdm

# ── Llegir API key des de Colab Secrets ──────────────────────────────────────
try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    import os
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise ValueError(
        "No s'ha trobat GROQ_API_KEY.\n"
        "Afegeix-la a Colab Secrets (icona 🔑) amb el nom GROQ_API_KEY."
    )

# ── Configuració ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CHUNKS_FILE     = Path("chunks.jsonl")   # Canvia a "chunks.jsonl" per al dataset complet
OUTPUT_FILE     = Path("train_data.jsonl")

GROQ_MODEL      = "llama-3.3-70b-versatile"      # Millor qualitat al tier gratuït
# GROQ_MODEL    = "llama-3.1-8b-instant"          # Alternativa més ràpida si arribes al límit

TARGET_EXAMPLES  = 10000       # Nombre d'exemples a generar (màx ~1000 recomanat per sessió)
MIN_CHUNK_CHARS  = 200         # Chunks massa curts no generen bones preguntes
MAX_RETRIES      = 3           # Reintents per chunk si falla la generació
REQUESTS_PER_MIN = 12          # Groq free: 30 req/min — deixem marge de seguretat
DRIVE_BACKUP_EVERY = 25        # Còpia de seguretat al Drive cada N exemples


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ets un expert jurídic especialitzat en la legislació del Principat d'Andorra.
La teva tasca és generar exemples d'entrenament per a un model d'IA jurídic en català.

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


# ── Funcions auxiliars ────────────────────────────────────────────────────────

def fix_encoding(s: str) -> str:
    """Corregeix el mojibake UTF-8/latin1 freqüent als fitxers BOPA."""
    if not s:
        return s
    repl = [
        ('\u00e2\u20ac\u2122', '\u2019'),  # â€™ → '
        ('\u00e2\u20ac\u0153', '\u201c'),  # â€œ → "
        ('\u00e2\u20ac\x9d',   '\u201d'),  # â€  → "
        ('\u00e2\u20ac\u201c', '\u2013'),  # â€" → –
        ('\u00e2\u20ac\u201d', '\u2014'),  # â€" → —
        ('\u00e2\u20ac\u00a2', '\u2022'),  # â€¢ → •
        ('\u00c2\u00b7',       '\u00b7'),  # Â· → ·
        ('\u00c2\u00bb',       '\u00bb'),  # Â» → »
        ('\u00c2\u00ab',       '\u00ab'),  # Â« → «
    ]
    for bad, good in repl:
        s = s.replace(bad, good)
    try:
        return s.encode('latin1').decode('utf-8')
    except Exception:
        return s


def load_chunks(path: Path) -> list[dict]:
    """Carrega chunks amb correcció d'encoding automàtica."""
    chunks = []
    # Suportar BOM (fitxers generats per PowerShell)
    encoding = 'utf-8-sig' if path.suffix == '.jsonl' else 'utf-8'
    with path.open("r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            c['text']         = fix_encoding(c.get('text', ''))
            c['source_title'] = fix_encoding(c.get('source_title', ''))
            chunks.append(c)
    logging.info(f"Chunks carregats: {len(chunks)}")
    return chunks


def filter_chunks(chunks: list[dict]) -> list[dict]:
    """Selecciona chunks aptes per generar preguntes de qualitat."""
    good = [
        c for c in chunks
        if c.get("n_chars", len(c["text"])) >= MIN_CHUNK_CHARS
        and "Article" in c["text"]
    ]
    logging.info(f"Chunks aptes (≥{MIN_CHUNK_CHARS} chars + Article): {len(good)} / {len(chunks)}")
    return good


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    """Carrega exemples ja generats per reprendre una execució interrompuda."""
    if not path.exists():
        return [], set()
    examples, seen_ids = [], set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                examples.append(ex)
                seen_ids.add(ex.get("source_chunk_id", ""))
    logging.info(f"Exemples existents (reprenen execució): {len(examples)}")
    return examples, seen_ids


# ── Google Drive ──────────────────────────────────────────────────────────────

DRIVE_BACKUP_DIR = None  # S'inicialitza a mount_drive()

def mount_drive() -> bool:
    """Munta Google Drive i crea la carpeta CLARA si no existeix."""
    global DRIVE_BACKUP_DIR
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        DRIVE_BACKUP_DIR = Path("/content/drive/MyDrive/CLARA")
        DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Google Drive muntat. Backup a: {DRIVE_BACKUP_DIR}")
        return True
    except Exception as e:
        logging.warning(f"No s'ha pogut muntar Drive: {e}\nEl progrés es desarà només en local.")
        return False


def backup_to_drive(n_examples: int):
    """Copia train_data.jsonl al Drive amb timestamp al nom."""
    if DRIVE_BACKUP_DIR is None:
        return
    try:
        import shutil
        dest = DRIVE_BACKUP_DIR / OUTPUT_FILE.name
        shutil.copy2(OUTPUT_FILE, dest)
        tqdm.write(f"  💾 Backup al Drive: {dest} ({n_examples} exemples)")
    except Exception as e:
        tqdm.write(f"  ⚠ Error al backup Drive: {e}")




client = Groq(api_key=GROQ_API_KEY)

# Comptador per al rate limiting
_request_times: list[float] = []

def _rate_limit():
    """Respectar el límit de REQUESTS_PER_MIN de Groq free tier."""
    now = time.time()
    # Eliminar timestamps de fa més d'un minut
    while _request_times and now - _request_times[0] > 60:
        _request_times.pop(0)
    if len(_request_times) >= REQUESTS_PER_MIN:
        sleep_time = 60 - (now - _request_times[0]) + 0.5
        if sleep_time > 0:
            logging.debug(f"Rate limit: esperant {sleep_time:.1f}s")
            time.sleep(sleep_time)
    _request_times.append(time.time())


def call_groq(chunk: dict) -> dict | None:
    """Crida Groq i retorna el parell instrucció-resposta parseajat."""
    prompt = USER_TEMPLATE.format(
        chunk_text=chunk["text"],
        source_title=chunk["source_title"],
    )

    _rate_limit()

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        raw_text = completion.choices[0].message.content or ""

        # Extreure el JSON (pot tenir text al voltant)
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            return None

        parsed = json.loads(json_match.group())

        if not parsed.get("instruction") or not parsed.get("output"):
            return None

        # Afegir metadades de traçabilitat
        parsed["input"]            = parsed.get("input", "")
        parsed["source_chunk_id"]  = chunk.get("chunk_id", "")
        parsed["source_title"]     = chunk.get("source_title", "")
        parsed["source_url"]       = chunk.get("source_url", "")

        return parsed

    except Exception as e:
        logging.debug(f"Error a call_groq: {e}")
        return None


# ── Pipeline principal ────────────────────────────────────────────────────────

def run():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{CHUNKS_FILE}'.\n"
            f"Puja el fitxer chunks_top1000.jsonl a Colab (icona 📁)."
        )

    # Muntar Drive per als backups automàtics
    mount_drive()

    chunks = load_chunks(CHUNKS_FILE)
    good   = filter_chunks(chunks)

    if not good:
        logging.warning("Cap chunk apte trobat. Revisa MIN_CHUNK_CHARS o el filtre 'Article'.")
        return

    # Reprendre execució prèvia si escau
    examples, seen_ids = load_existing(OUTPUT_FILE)
    remaining_target   = TARGET_EXAMPLES - len(examples)

    if remaining_target <= 0:
        logging.info(f"Ja tens {len(examples)} exemples. TARGET_EXAMPLES assolit.")
        return

    random.seed(42)
    random.shuffle(good)
    pending = [c for c in good if c.get("chunk_id", "") not in seen_ids]

    logging.info(f"Chunks pendents: {len(pending)} | Objectiu restant: {remaining_target}")
    logging.info(f"Model: {GROQ_MODEL} | Velocitat estimada: ~2s/exemple")
    logging.info(f"Temps estimat: ~{remaining_target * 2 // 60} min")

    out_file = OUTPUT_FILE.open("a", encoding="utf-8")
    errors   = 0
    generated_this_run = 0

    with tqdm(total=remaining_target, desc="Generant exemples", unit="ex") as pbar:
        for chunk in pending:
            if generated_this_run >= remaining_target:
                break

            success = False
            for attempt in range(MAX_RETRIES):
                result = call_groq(chunk)
                if result:
                    examples.append(result)
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
                    pbar.update(1)
                    pbar.set_postfix({
                        "errors": errors,
                        "total":  len(examples),
                        "model":  GROQ_MODEL.split("-")[2],  # "70b" o "8b"
                    })
                    generated_this_run += 1
                    success = True

                    # Backup al Drive cada DRIVE_BACKUP_EVERY exemples
                    if generated_this_run % DRIVE_BACKUP_EVERY == 0:
                        out_file.flush()
                        backup_to_drive(len(examples))

                    break
                time.sleep(2 ** attempt)  # Backoff exponencial: 1s, 2s, 4s

            if not success:
                errors += 1
                tqdm.write(f"  ⚠ Error: {chunk['source_title'][:60]}")

    out_file.close()

    # Backup final al Drive
    backup_to_drive(len(examples))

    # ── Estadístiques finals ──────────────────────────────────────────────────
    logging.info("=" * 55)
    logging.info(f"Exemples generats:     {len(examples)}")
    logging.info(f"Errors / omesos:       {errors}")
    logging.info(f"Dataset desat a:       {OUTPUT_FILE}")
    logging.info("=" * 55)
    logging.info("Proper pas: descarrega train_data.jsonl i puja'l")
    logging.info("a Google Colab per executar fase3_finetune_colab.ipynb")

    # Descarregar automàticament a Colab
    try:
        from google.colab import files
        files.download(str(OUTPUT_FILE))
        logging.info("Descàrrega iniciada automàticament.")
    except Exception:
        logging.info(f"Descarrega manualment el fitxer '{OUTPUT_FILE}' des del panell 📁 de Colab.")


if __name__ == "__main__":
    run()
