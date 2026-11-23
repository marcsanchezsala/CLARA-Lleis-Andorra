
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

src = r"llama-clara-v2-hf"
dst = r"llama-clara-v2-fp16"

print("Carregant model BNB 4-bit (pot trigar uns minuts)...")
model = AutoModelForCausalLM.from_pretrained(
    src,
    device_map="cpu",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(src)

print(f"Desant model fp16 a {dst}...")
model.save_pretrained(dst, safe_serialization=True)
tokenizer.save_pretrained(dst)
print("Model desquantitzat desat correctament.")
