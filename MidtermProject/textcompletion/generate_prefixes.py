from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
import random
import time


cache_dirmodel="./models/models--meta-llama--Llama-2-7b-hf--0/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
cache_dirtokenizer="./models/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
tokenizer = AutoTokenizer.from_pretrained(cache_dirtokenizer)
model = AutoModelForCausalLM.from_pretrained(cache_dirmodel)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_prefixes(model, tokenizer, num_prefixes=10, max_length=7):
    prefixes = []
    possible_starters = ["While", "If", "However", "Although", "Since", "In order to", "Yesterday", "Once", "Because"]

    for i in range(num_prefixes):
        prompt = random.choice(possible_starters)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
        
        prefix = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prefixes.append(prefix)
        print(len(prefixes))
    
    return prefixes

num_prefixes = 3000
prefixes = generate_prefixes(model, tokenizer, num_prefixes=num_prefixes)
df = pd.DataFrame(prefixes, columns=["Prefix"])
df.to_csv("llama_prefixes.csv", index=False)
print(f"Generated {num_prefixes} prefixes and saved them to llama_prefixes.csv")
