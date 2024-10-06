from openai import OpenAI
import pandas as pd
client = OpenAI()

start_prompt = "Please complete this sentence as briefly as possible: "
model = "gpt-3.5-turbo-instruct"

def read_prefixes_in_batches(file_path, batch_size=3):
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        prefix_list = chunk['Prefix'].tolist()
        yield prefix_list

def make_batch_api_calls(file_path):
    for batch in read_prefixes_in_batches(file_path):
        for prefix in batch:
            completion = client.completions.create(
                model=model,
                prompt=f"{start_prompt} {prefix}",
                max_tokens=50, 
                temperature=0.5
            )

            suffix_text = completion.choices[0].text.replace("\n", " ").strip()
            stop_tokens = ['. ', '! ', '? ']
            stop_index = min((suffix_text.find(token) for token in stop_tokens if suffix_text.find(token) != -1), default=len(suffix_text))
            suffix_text = suffix_text[:stop_index + 1]
            print(prefix + "|" + suffix_text,  flush=True)

make_batch_api_calls("llama_prefixes.csv")