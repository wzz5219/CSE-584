import google.generativeai as genai
import os
import sys
import time
import pandas as pd

genai.configure(api_key=os.environ["API_KEY"])
start_prompt = "i give you incomplete text. just finish it as one complete sentence as briefly as possible:"

model = genai.GenerativeModel("gemini-1.5-flash")

def read_prefixes_in_batches(file_path, start_row, batch_size=10):
    for chunk in pd.read_csv(file_path, chunksize=batch_size,skiprows=range(1, start_row)):
        prefix_list = chunk['Prefix'].tolist()
        yield prefix_list

def make_batch_api_calls(file_path, start_row):
    for batch in read_prefixes_in_batches(file_path, start_row):
        for prefix in batch:
            response = model.generate_content(
                f"{start_prompt} {prefix}",
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=50,
                    temperature=0.5,
                )
            )

            if hasattr(response, 'text'):
                suffix_text = response.text
            else:
                suffix_text = "text raised flag." #gemini sometimes do not generate output.
            if suffix_text.startswith(prefix):
                suffix_text = suffix_text[len(prefix):]
            suffix_text = suffix_text.replace("\n", " ").strip()
            stop_tokens = ['. ', '! ', '? ']
            stop_index = min(
                (suffix_text.find(token) for token in stop_tokens if suffix_text.find(token) != -1),
                default=len(suffix_text)
            )
            suffix_text = suffix_text[:stop_index + 1]
            print(prefix + "|" + suffix_text, flush=True)

            time.sleep(4)    

make_batch_api_calls("llama_prefixes.csv", int(sys.argv[1]))
