import os
import time
from mistralai import Mistral
import pandas as pd

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small"

client = Mistral(api_key=api_key)

start_prompt = "Please complete this truncated sentence as briefly as possible starting with the given prefix: "

def read_prefixes_in_batches(file_path, batch_size=10):
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        prefix_list = chunk['Prefix'].tolist()
        yield prefix_list

def make_batch_api_calls(file_path):
    for batch in read_prefixes_in_batches(file_path):
        for prefix in batch:
            completion = client.chat.complete(
                model= model,
                messages = [
                    {
                        "role": "user",
                        "content": f"{start_prompt} {prefix}",
                    },
                ],
                max_tokens=50,  
                temperature=0.7
            )

            suffix_text = completion.choices[0].message.content
            if not(prefix.startswith('"') or prefix.startswith("'")):
                suffix_text = suffix_text.lstrip('"').lstrip("'")
            if suffix_text.startswith(prefix):
                suffix_text = suffix_text[len(prefix):]
            suffix_text = suffix_text.replace("\n", " ").strip()
            stop_tokens = ['. ', '! ', '? ']  
            stop_index = min((suffix_text.find(token) for token in stop_tokens if suffix_text.find(token) != -1), default=len(suffix_text))
            suffix_text = suffix_text[:stop_index + 1]
            print(prefix + "|" + suffix_text,  flush=True)
            time.sleep(2)
        
make_batch_api_calls("llama_prefixes.csv")
