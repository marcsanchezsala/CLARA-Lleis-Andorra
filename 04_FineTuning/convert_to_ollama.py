"""
CLARA — Conversió local: HuggingFace → GGUF → Ollama
=====================================================
Requisits previs:
  - Python 3.10+
  - Ollama instal·lat (https://ollama.com)
  - Carpeta llama-b8249-bin-win-cpu-x64 present al mateix directori
  - ~20 GB d'espai lliure en disc

Execució:
  python convert_to_ollama.py
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ─── CONFIGURACIÓ ─────────────────────────────────────────────────────────────
HF_REPO       = "MSS136/llama-clara-v2"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")   # <-- opcional si el model és públic, o defineix la variable d'entorn HF_TOKEN
MODEL_DIR     = Path("llama-clara-v2-hf")
MODEL_FP16_DIR = Path("llama-clara-v2-fp16")      # directori temporal desquantitzat
GGUF_F16      = Path("llama-clara-v2-f16.gguf")
GGUF_Q4       = Path("llama-clara-v2-Q4_K_M.gguf")
OLLAMA_NAME   = "clara"

# Carpeta dels binaris de llama.cpp que ja tens
LLAMA_BIN_DIR = Path("llama-b8249-bin-win-cpu-x64")
# ──────────────────────────────────────────────────────────────────────────────


def run(cmd, cwd=None):
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"❌ Error executant: {cmd}")
        sys.exit(1)


def step(n, total, msg):
    print(f"\n{'='*60}")
    print(f"  PAS {n}/{total} — {msg}")
    print(f"{'='*60}")


def download_file(url: str, dest: Path):
    print(f"⬇️  Descarregant {dest.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024 * 64
        count = 0
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                count += 1
                if total_size > 0:
                    pct = min(count * block_size / total_size * 100, 100)
                    mb = count * block_size / 1e6
                    print(f"\r   {pct:.1f}%  ({mb:.0f} / {total_size/1e6:.0f} MB)", end="", flush=True)
    print()


# ── Verificar que la carpeta de binaris existeix ──────────────────────────────
if not LLAMA_BIN_DIR.exists():
    print(f"❌ No s'ha trobat la carpeta: {LLAMA_BIN_DIR}")
    sys.exit(1)

candidates = list(LLAMA_BIN_DIR.rglob("llama-quantize.exe"))
if not candidates:
    print(f"❌ No s'ha trobat llama-quantize.exe dins {LLAMA_BIN_DIR}")
    for f in LLAMA_BIN_DIR.rglob("*.exe"):
        print(f"   {f}")
    sys.exit(1)

quantize_bin = candidates[0]
print(f"✅ llama-quantize trobat: {quantize_bin}")


# ── PAS 1: Instal·lar dependències Python ─────────────────────────────────────
step(1, 5, "Instal·lar dependències Python")
run("pip install huggingface_hub numpy transformers sentencepiece protobuf gguf accelerate")
run("pip install torch --index-url https://download.pytorch.org/whl/cpu")
# bitsandbytes >= 0.46.1 ja te suport Windows natiu (no cal wheel de tercers)
run('pip install "bitsandbytes>=0.46.1"')


# ── PAS 2: Descarregar model de HuggingFace ───────────────────────────────────
step(2, 5, f"Descarregar model des de {HF_REPO}")
if not MODEL_DIR.exists():
    from huggingface_hub import snapshot_download
    print(f"Descarregant a {MODEL_DIR} ...")
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN or None,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"✅ Model descarregat a {MODEL_DIR}")
else:
    print(f"ℹ️  {MODEL_DIR} ja existeix, saltant descàrrega.")


# ── PAS 3: Descarregar script de conversió HF → GGUF ─────────────────────────
step(3, 5, "Preparar script de conversió")

convert_script = LLAMA_BIN_DIR / "convert_hf_to_gguf.py"
if not convert_script.exists():
    for branch in ["b8249", "master"]:
        url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{branch}/convert_hf_to_gguf.py"
        print(f"⬇️  Provant branch '{branch}'...")
        try:
            download_file(url, convert_script)
            print(f"✅ Script descarregat des de '{branch}'.")
            break
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            if convert_script.exists():
                convert_script.unlink()
    else:
        print("❌ No s'ha pogut descarregar convert_hf_to_gguf.py")
        sys.exit(1)
else:
    print(f"ℹ️  {convert_script} ja existeix.")


# ── PAS 4: Desquantitzar BNB + Convertir a GGUF ──────────────────────────────
step(4, 5, "Desquantitzar BNB → fp16 → GGUF → Q4_K_M")

# El model a HF està en format bitsandbytes 4-bit (BNB).
# convert_hf_to_gguf.py no el suporta directament, cal desquantitzar primer.

if not GGUF_F16.exists():

    # 4a. Desquantitzar BNB → float16
    if not MODEL_FP16_DIR.exists():
        print("Desquantitzant el model BNB 4-bit → float16...")
        print("(Necessita ~16 GB de RAM, pot trigar 10-20 minuts)")

        dequant_script = Path("_dequant_bnb.py")
        dequant_script.write_text(
            f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src = r"{MODEL_DIR}"
dst = r"{MODEL_FP16_DIR}"

print("Carregant model BNB 4-bit (pot trigar uns minuts)...")
# Carreguem amb bitsandbytes per desquantitzar; save_pretrained escriu en fp16
model = AutoModelForCausalLM.from_pretrained(
    src,
    device_map="cpu",
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(src)

print(f"Desant model fp16 a {{dst}}...")
# save_pretrained sense quantitzar → escriu els pesos en float16 pur
model = model.dequantize()
model.save_pretrained(dst, safe_serialization=True)
tokenizer.save_pretrained(dst)
print("Model desquantitzat desat correctament.")
""",
            encoding="utf-8",
        )

        run(f'python "{dequant_script}"')
        dequant_script.unlink()
        print(f"✅ Model fp16 desat a {MODEL_FP16_DIR}")
    else:
        print(f"ℹ️  {MODEL_FP16_DIR} ja existeix, saltant desquantització.")

    # 4b. Convertir fp16 → GGUF F16
    print("Convertint fp16 → GGUF F16 (pot trigar 5-10 minuts)...")
    run(f'python "{convert_script}" "{MODEL_FP16_DIR}" --outtype f16 --outfile "{GGUF_F16}"')
    print(f"✅ GGUF F16 creat: {GGUF_F16} ({GGUF_F16.stat().st_size / 1e9:.1f} GB)")

else:
    print(f"ℹ️  {GGUF_F16} ja existeix, saltant conversió.")

# 4c. Quantitzar a Q4_K_M
if not GGUF_Q4.exists():
    print("Quantitzant a Q4_K_M (pot trigar 5-10 minuts)...")
    run(f'"{quantize_bin}" "{GGUF_F16}" "{GGUF_Q4}" Q4_K_M')
    print(f"✅ GGUF Q4_K_M creat: {GGUF_Q4}")

    # Eliminar intermedis per estalviar espai
    for tmp in [GGUF_F16, MODEL_FP16_DIR]:
        if isinstance(tmp, Path) and tmp.exists():
            if tmp.is_file():
                size = tmp.stat().st_size / 1e9
                print(f"🗑️  Eliminant {tmp.name} ({size:.1f} GB)...")
                tmp.unlink()
            else:
                import shutil
                size = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file()) / 1e9
                print(f"🗑️  Eliminant {tmp.name}/ ({size:.1f} GB)...")
                shutil.rmtree(tmp)
            print("   Eliminat.")
else:
    print(f"ℹ️  {GGUF_Q4} ja existeix, saltant quantització.")

size_gb = GGUF_Q4.stat().st_size / 1e9
print(f"\n📦 Mida final del model: {size_gb:.1f} GB")


# ── PAS 5: Crear Modelfile i carregar a Ollama ────────────────────────────────
step(5, 5, "Carregar a Ollama")

gguf_abs = GGUF_Q4.resolve()

modelfile_content = (
    f"FROM {gguf_abs}\n\n"
    "PARAMETER temperature 0.1\n"
    "PARAMETER top_p 0.9\n"
    "PARAMETER repeat_penalty 1.1\n"
    "PARAMETER num_ctx 4096\n\n"
    'SYSTEM """Ets CLARA, un expert jurídic especialitzat en la legislació del '
    "Principat d'Andorra. Respon sempre en català correcte. Cita sempre l'article "
    'i la llei corresponents quan sigui possible. Si no tens la informació al context, '
    'indica-ho explícitament."""\n'
)

modelfile_path = Path("Modelfile")
modelfile_path.write_text(modelfile_content, encoding="utf-8")
print("✅ Modelfile creat")

run(f'ollama create {OLLAMA_NAME} -f "{modelfile_path}"')
print(f"\n✅ Model '{OLLAMA_NAME}' creat a Ollama!")


# ── RESUM FINAL ───────────────────────────────────────────────────────────────
print(f"""
{'='*60}
  PROCES COMPLETAT
{'='*60}

  Per provar CLARA directament:
    ollama run {OLLAMA_NAME}

  Per arrencar la UI:
    ollama serve
    streamlit run app.py
{'='*60}
""")


'''
python -c "
import shutil
from pathlib import Path

src = Path('llama-clara-v2-hf')
dst = Path('llama-clara-v2-fp16')

# Copiar tots els fitxers del tokenitzador que falten
tokenizer_files = [
    'tokenizer.json',
    'tokenizer_config.json', 
    'tokenizer.model',
    'special_tokens_map.json',
    'chat_template.jinja',
]
for fname in tokenizer_files:
    s = src / fname
    if s.exists() and not (dst / fname).exists():
        shutil.copy2(s, dst / fname)
        print(f'Copiat: {fname}')
    elif (dst / fname).exists():
        print(f'Ja existeix: {fname}')
    else:
        print(f'No trobat a src: {fname}')
print('Fet.')
"





python -c "
import urllib.request
from pathlib import Path

dest = Path('llama-b8249-bin-win-cpu-x64/convert_hf_to_gguf.py')
url = 'https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py'
print('Descarregant convert_hf_to_gguf.py des de master...')
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
    f.write(r.read())
print('Fet.')
"
'''