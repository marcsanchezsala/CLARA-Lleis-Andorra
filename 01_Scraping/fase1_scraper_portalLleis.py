"""
CLARA — Corpus Legal Andorrà amb Recuperació Augmentada
Fase 1: Scraping del Portal Jurídic del Principat d'Andorra

Font:   https://portaljuridicandorra.ad
Motor:  Tiki Wiki CMS amb Trackers (HTML estàtic, no SPA)

Estratègia de scraping:
  1. IndexPerAnys: GET amb paràmetre d'any → taula de lleis
  2. Filtrar per situació:
       VIGENT (verd)       → text consolidat en vigor          ✅ INCLÒS
       VACATIO LEGIS (groc)→ aprovat però pendent d'entrar en vigor ❌ EXCLÒS
       FITXA (blau dl490)  → norma sense seguiment posterior   ✅ INCLÒS (configurable)
       DEROGAT (vermell)   → no vigent                         ❌ EXCLÒS
  3. Per cada llei: descarregar pàgina de detall → extreure text
  4. Netejar i desar .txt per al chunking (fase1_chunker.py)

Ús:
    python fase1_scraper_portalLleis.py

Configura MAX_DOCS_TOTAL = 20  per fer proves ràpides.
"""

import time
import random
import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Directoris (SEMPRE PRIMER, abans del FileHandler) ─────────────────────────

for _d in [Path("data"), Path("data/raw_html"), Path("data/clean_texts")]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL  = "https://portaljuridicandorra.ad"
INDEX_URL = f"{BASE_URL}/tiki-index.php"
RAW_DIR   = Path("data/raw_html")
CLEAN_DIR = Path("data/clean_texts")
META_FILE = Path("data/metadata.jsonl")

# ── Configuració (editable) ───────────────────────────────────────────────────

# Anys a scrapejar. [] = tots (detecció automàtica des de l'índex).
# Per proves: [2025, 2026]
ANYS_A_SCRAPEJAR: list[int] = []

# ── Situacions a incloure ──────────────────────────────────────────────────────
INCLOU_FITXES: bool = True   # ← Canvia a False si no vols les fitxes blaves

def _build_situacions() -> set[str]:
    s = {"vigent"}
    if INCLOU_FITXES:
        s.add("fitxa")
    return s

SITUACIONS_INCLOSES: set[str] = _build_situacions()

MAX_DOCS_TOTAL = 0    # 0 = tots; ex. 30 per proves ràpides
RATE_MIN = 0.2        # segons entre peticions HTTP
RATE_MAX = 1

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (compatible; ClaraLegalBot/1.0; +research)",
    "Accept":          "text/html,application/xhtml+xml",
    "Accept-Language": "ca,ca-AD;q=1.0,es;q=0.7",
}

TRACKER_FIELD_ANY = "f_1"   # Nom del camp "Any" al tracker de Tiki Wiki


# ── Sessió HTTP ───────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update(HEADERS)


def polite_get(url: str, params: dict = None, retries: int = 3) -> requests.Response | None:
    """GET amb rate limiting cortès i reintents automàtics."""
    time.sleep(random.uniform(RATE_MIN, RATE_MAX))
    for attempt in range(1, retries + 1):
        try:
            r = _session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            logging.warning(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    logging.error(f"  Abandono {url}")
    return None


# ── Pas 1: Índex de lleis per any ────────────────────────────────────────────

def get_index_page(year: int) -> requests.Response | None:
    params = {
        "page":           "IndexPerAnys",
        "trackerFilter1": "1",
        TRACKER_FIELD_ANY: str(year),
    }
    r = polite_get(INDEX_URL, params=params)
    if r and "trobats" in r.text.lower():
        return r

    params2 = {
        "page":           "IndexPerAnys",
        "trackerFilter1": "1",
        "f_any":          str(year),
    }
    r2 = polite_get(INDEX_URL, params=params2)
    if r2 and "trobats" in r2.text.lower():
        return r2

    return r


def detect_tracker_field(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["input", "select"]):
        name = tag.get("name", "").lower()
        label_text = ""
        tag_id = tag.get("id", "")
        if tag_id:
            label = soup.find("label", {"for": tag_id})
            if label:
                label_text = label.get_text().lower()
        if any(x in name + label_text for x in ["any", "year", "año", "data"]):
            logging.info(f"  Camp 'Any' detectat: name={tag.get('name')!r}")
            return tag.get("name", TRACKER_FIELD_ANY)

    for form in soup.find_all("form"):
        selects = form.find_all("select")
        if selects:
            name = selects[0].get("name", TRACKER_FIELD_ANY)
            logging.info(f"  Camp 'Any' (fallback): {name!r}")
            return name

    return TRACKER_FIELD_ANY


def get_available_years(html: str) -> list[int]:
    soup = BeautifulSoup(html, "html.parser")
    for sel in soup.find_all("select"):
        opts = sel.find_all("option")
        years = []
        for o in opts:
            txt = o.get_text(strip=True)
            val = o.get("value", txt)
            for candidate in (val, txt):
                if re.match(r"^(19|20)\d{2}$", candidate.strip()):
                    years.append(int(candidate.strip()))
        if years:
            logging.info(f"  Anys al selector: {sorted(set(years))}")
            return sorted(set(years))
    return []


# ── Pas 2: Parseig de la taula de lleis ──────────────────────────────────────

def classify_situation(cell_html: str) -> str:
    ICON_MAP = {
        "dl484": "vigent",
        "dl485": "vigent",
        "dl486": "vacatio",
        "dl487": "derogat",
        "dl488": "derogat",
        "dl490": "fitxa",
        "dl491": "derogat",
    }

    soup = BeautifulSoup(cell_html, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src", "").strip().lower()
        for dl_id, situacio in ICON_MAP.items():
            if dl_id in src:
                return situacio

    for img in soup.find_all("img"):
        src = img.get("src", "").lower()
        alt = img.get("alt", "").lower()
        if any(x in src + alt for x in ["vigent", "green", "activ"]):
            return "vigent"
        if any(x in src + alt for x in ["vacatio", "yellow", "warn"]):
            return "vacatio"
        if any(x in src + alt for x in ["derogat", "red", "dero", "modif"]):
            return "derogat"
        if any(x in src + alt for x in ["fitxa", "blue", "ficha"]):
            return "fitxa"

    return "desconegut"


def parse_laws_table(html: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    count_text = soup.find(string=re.compile(r"[Íi]tems\s+trobats", re.I))
    if count_text:
        m = re.search(r"\d+", count_text)
        if m:
            logging.info(f"  Ítems trobats: {m.group()}")

    table = None
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in t.find_all("th")]
        if any(x in " ".join(headers) for x in ["títol", "titol", "title", "situació"]):
            table = t
            break

    if not table:
        tables = soup.find_all("table")
        if tables:
            table = max(tables, key=lambda t: len(t.find_all("tr")))

    if not table:
        logging.warning(f"  Any {year}: cap taula de resultats trobada")
        Path(f"data/debug_{year}.html").write_bytes(html.encode("utf-8", "replace"))
        logging.info(f"  HTML desat a data/debug_{year}.html per diagnosi")
        return []

    headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
    logging.debug(f"  Columnes detectades: {headers}")

    def col_idx(candidates: list[str], default: int) -> int:
        for cand in candidates:
            for i, h in enumerate(headers):
                if cand in h:
                    return i
        return default

    i_sit   = col_idx(["situació", "situacio", "estat", "status"], 0)
    i_title = col_idx(["títol", "titol", "title", "nom"], 1)
    i_ddata = col_idx(["data doc", "data d", "document"], 2)
    i_dpub  = col_idx(["publicació", "publicacio", "pub"], 3)
    i_dvig  = col_idx(["vigor", "vigència", "vigencia", "vigent"], 4)

    laws = []
    rows = table.find_all("tr")[1:]

    if rows:
        diag_path = Path(f"data/debug_row_{year}.html")
        diag_html = "\n\n<!-- === FILA === -->\n\n".join(str(r) for r in rows[:3])
        diag_path.write_text(diag_html, encoding="utf-8")
        logging.debug(f"  HTML de les primeres 3 files desat a {diag_path} per diagnosi")

    for row in rows:
        cols = row.find_all(["td", "th"])
        if len(cols) < 2:
            continue

        sit_html  = str(cols[i_sit]) if i_sit < len(cols) else ""
        situacio  = classify_situation(sit_html)

        title_cell = cols[i_title] if i_title < len(cols) else cols[-1]
        link = title_cell.find("a", href=True)
        if not link:
            continue
        title   = link.get_text(strip=True)
        href    = link["href"]
        url_llei = href if href.startswith("http") else urljoin(BASE_URL, href)

        def get_date(idx: int) -> str:
            if idx < len(cols):
                t = cols[idx].get_text(strip=True)
                if re.match(r"\d{2}/\d{2}/\d{4}", t):
                    return t
            row_text = row.get_text(" ")
            dates    = re.findall(r"\d{2}/\d{2}/\d{4}", row_text)
            return dates[0] if dates else ""

        laws.append({
            "title":       title,
            "url":         url_llei,
            "year":        year,
            "date_doc":    get_date(i_ddata),
            "date_pub":    get_date(i_dpub),
            "date_vigor":  get_date(i_dvig),
            "situacio":    situacio,
        })

    return laws


# ── Pas 3: Descarregar i extreure text de cada llei ──────────────────────────

def extract_text_from_page(html_bytes: bytes) -> str:
    """
    Extreu el text legal eliminant els índexs automàtics (TOC) i capçaleres
    de Tiki Wiki abans de convertir a text. Unwraps inline tags per evitar
    salts de línia artificials en paraules com <i>ter</i>.
    """
    for enc in ("utf-8", "iso-8859-1", "windows-1252"):
        try:
            html = html_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        html = html_bytes.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")

    # 1. Eliminar etiquetes no textuals i elements d'UI
    for tag in soup(["script", "style", "nav", "header", "footer",
                      "aside", "iframe", "noscript", "meta", "link",
                      "form", "button"]):
        tag.decompose()

    # 1b. Unir contingut d'etiquetes d'estil en línia per evitar trencar paraules/frases
    for tag in soup.find_all(["i", "b", "em", "strong", "span", "sub", "sup"]):
        tag.unwrap()

    # 2. Eliminar índexs autogenerats (TOC / Table of Contents)
    for toc_tag in soup.find_all(class_=re.compile(r"toc|autotoc|tiki-toc|index_list", re.I)):
        toc_tag.decompose()
    for toc_tag in soup.find_all(id=re.compile(r"toc|autotoc|tiki-toc", re.I)):
        toc_tag.decompose()

    # 3. Localitzar el bloc principal
    content = (
        soup.find("div", class_=re.compile(r"^wiki.?page$|^wiki.?content$", re.I))
        or soup.find("div", id=re.compile(r"wikicontent|wiki-content|content", re.I))
        or soup.find("div", class_="content")
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"text|body|article", re.I))
    )

    if content:
        return content.get_text(separator="\n")

    body = soup.find("body")
    if body:
        for sidebar in body.find_all("div", class_=re.compile(r"side|nav|menu|col-1|lateral", re.I)):
            sidebar.decompose()
        return body.get_text(separator="\n")

    return soup.get_text(separator="\n")

def clean_legal_text(text: str) -> str:
    """
    Neteja el text legal:
    - Elimina caràcters de control
    - Normalitza tipografia catalana
    - Elimina etiquetes {includes page=...} de Tiki Wiki
    - Elimina elements de UI i índexs autogenerats
    - Elimina la cua administrativa (signatures, fórmules finals i XXXXXXXX)
    - Repara sufixos llatins (bis, ter, quater, etc.) trencats per salts de línia
    - Normalitza espais i elimina línies blanques múltiples
    """
    # 1. Caràcters de control
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Tipografia catalana/estàndard
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u00a0": " ", "\u2026": "...",
        "\u00b7": "·",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 3. Eliminar etiquetes d'inclusió de Tiki Wiki {includes page=...}
    text = re.sub(r"\{includes\s+page=[^}]+\}", "", text, flags=re.I)

    # 4. Eliminar blocs d'índex autogenerats i UI
    text = re.sub(r"\[Mostra/Amaga\]|\[Amaga\]|\[Mostra\]", "", text, flags=re.I)
    text = re.sub(r"\[Show/Hide\]|\[Show\]|\[Hide\]", "", text, flags=re.I)

    ui_patterns = [
        r"^(Menú|Menu|Cercar|Cerca|BOPA|Consell General|Portal jurídic)\s*$",
        r"^(Objecte i contingut|Avis legal|Manual d.ús|Contacte)\s*$",
        r"^(Inici|Amunt|Edita|Historial|Imprimir)\s*$",
        r"^(Anterior|Següent|Pàgina \d+)\s*$",
        r"^\s*(VIGENT|DEROGAT|VACATIO|MODIFICAT|FITXA)\s*$",
        r"^Índex\s*$",
        r"^Índex\s*\[.*?\]\s*$",
    ]
    for pat in ui_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.I)

    # Números de pàgina aïllats
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # 5. ELIMINAR CUA ADMINISTRATIVA (Signatures, decrets finals i XXXXXXXX)
    text = re.sub(
        r"Cosa que es fa pública per a coneixement general\..*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"\n\s*Andorra la Vella,\s*\d+.*?(?:Cap de Govern|Ministr\w+).*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r"(?:Aprovat per|Inclou modificacions de|Derogat per):\s*(?:X{3,}|[\s\n]*)*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 6. REPARACIÓ DE SUFIXOS LLATINS (bis, ter, quater, etc.)
    latin_suffixes = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)"
    # Reconnectar número amb sufix (ex: "26\nter" -> "26 ter")
    text = re.sub(
        rf"(\d+)\s*\n\s*({latin_suffixes})\b",
        r"\1 \2",
        text,
        flags=re.IGNORECASE
    )
    # Reconnectar sufix amb parèntesi/puntuació de tancament (ex: "ter\n)" -> "ter)")
    text = re.sub(
        rf"\b({latin_suffixes})\s*\n\s*([\)\,\.])",
        r"\1\2",
        text,
        flags=re.IGNORECASE
    )

    # 7. NORMALITZACIÓ D'ESPAIS I LÍNIES BLANQUES
    # A) Netejar espais al final de cada línia (converteix línies amb espais buits en \n netes)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # B) Reduir espais dobles o múltiples horitzontals a un sol espai
    text = re.sub(r"[ \t]{2,}", " ", text)

    # C) Col·lapsar 3 o més salts de línia consecutius a exactament 2 (\n\n = 1 línia en blanc entre paràgrafs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def download_law(law: dict) -> tuple[str | None, Path]:
    safe = re.sub(r"[^\w\-]", "_", law["title"][:60])
    subdir = "fitxes" if law.get("situacio") == "fitxa" else str(law["year"])
    html_path = RAW_DIR / subdir / f"{safe}.html"

    if html_path.exists():
        logging.debug(f"  Cache: {html_path.name}")
        content = html_path.read_bytes()
    else:
        r = polite_get(law["url"])
        if not r:
            return None, html_path
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_bytes(r.content)
        content = r.content
        tag = "[FITXA]" if law.get("situacio") == "fitxa" else ""
        logging.info(f"  ↓ {tag} {html_path.name} ({len(content)//1024} KB)")

    raw_text = extract_text_from_page(content)
    clean    = clean_legal_text(raw_text)
    return clean, html_path


# ── Pipeline principal ────────────────────────────────────────────────────────

def run():
    logging.info("=" * 60)
    logging.info("CLARA — Scraper del Portal Jurídic d'Andorra  [v2]")
    logging.info(f"Font:      {BASE_URL}")
    logging.info(f"Situacions incloses: {SITUACIONS_INCLOSES}")
    logging.info(f"  INCLOU_FITXES = {INCLOU_FITXES}")
    logging.info(f"MAX_DOCS:  {MAX_DOCS_TOTAL or 'tots'}")
    logging.info("=" * 60)

    logging.info("Pas 0: Diagnosi inicial...")
    r0 = polite_get(INDEX_URL, params={"page": "IndexPerAnys"})
    if not r0:
        logging.error("No s'ha pogut connectar al portal. Verifica la connexió.")
        return

    Path("data/index_raw.html").write_bytes(r0.content)
    logging.info("  HTML inicial desat a data/index_raw.html")

    field_any = detect_tracker_field(r0.text)
    logging.info(f"  Nom del camp 'Any' detectat: {field_any!r}")

    available_years = get_available_years(r0.text)
    if not available_years:
        logging.warning("  No s'han detectat anys. Usant rang 2000-2026 per defecte.")
        available_years = list(range(2000, 2027))

    years = sorted(ANYS_A_SCRAPEJAR or available_years, reverse=True)
    logging.info(f"  Anys a processar: {years}\n")

    logging.info("Pas 1: Recollint índex de lleis per any...")
    all_laws: list[dict] = []

    for year in years:
        params = {
            "page":           "IndexPerAnys",
            "trackerFilter1": "1",
            field_any:        str(year),
        }
        r = polite_get(INDEX_URL, params=params)
        if not r:
            continue

        laws = parse_laws_table(r.text, year)
        incloses = [l for l in laws if l["situacio"] in SITUACIONS_INCLOSES]
        fitxes_n = sum(1 for l in incloses if l["situacio"] == "fitxa")
        vigents_n = sum(1 for l in incloses if l["situacio"] == "vigent")

        logging.info(
            f"  {year}: {len(laws):>3} total  |  "
            f"{vigents_n:>3} vigents  |  {fitxes_n:>3} fitxes  → {len(incloses):>3} incloses"
        )

        all_laws.extend(incloses)

        if MAX_DOCS_TOTAL and len(all_laws) >= MAX_DOCS_TOTAL:
            all_laws = all_laws[:MAX_DOCS_TOTAL]
            logging.info(f"  Límit MAX_DOCS_TOTAL={MAX_DOCS_TOTAL} assolit.")
            break

    vigents_total = sum(1 for l in all_laws if l["situacio"] == "vigent")
    fitxes_total  = sum(1 for l in all_laws if l["situacio"] == "fitxa")
    logging.info(f"\nTotal lleis a descarregar: {len(all_laws)}")
    logging.info(f"  Vigents: {vigents_total}  |  Fitxes (blaves): {fitxes_total}")

    if not all_laws:
        logging.error("0 lleis trobades.")
        return

    logging.info("\nPas 2: Descarregant lleis...")
    results: list[dict] = []

    for i, law in enumerate(all_laws, 1):
        if i % 25 == 0:
            logging.info(f"  Progrés: {i}/{len(all_laws)} lleis")

        clean, html_path = download_law(law)

        if not clean or len(clean) < 150:
            logging.debug(f"  Text massa curt o buit: {law['title'][:60]}")
            continue

        subdir = "fitxes" if law.get("situacio") == "fitxa" else str(law["year"])
        safe = re.sub(r"[^\w\-]", "_", law["title"][:60])
        clean_path = CLEAN_DIR / subdir / f"{safe}.txt"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_text(clean, encoding="utf-8")

        results.append({
            **law,
            "local_path": str(html_path),
            "clean_path": str(clean_path),
            "n_chars":    len(clean),
        })

    with META_FILE.open("w", encoding="utf-8") as f:
        for doc in results:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    from collections import Counter
    logging.info("\n" + "=" * 60)
    logging.info(f"Lleis processades:  {len(results)}")
    logging.info(f"Metadades:          {META_FILE}")
    logging.info(f"HTMLs:              {RAW_DIR}")
    logging.info(f"Textos nets:        {CLEAN_DIR}")


if __name__ == "__main__":
    run()