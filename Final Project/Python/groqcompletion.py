import pandas as pd
import csv
from groq import Groq
import os

client = Groq(
    api_key="",
)

suffix_prompt = ". Please make sure the question is valid beforehand."
suffix_prompt_1 = ""


def read_prefixes_in_batches(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Read each row
        for row in reader:
            for cell in row:
                # Strip quotes if necessary
                cell_content = cell#.strip('"')
                yield cell_content
                #print(cell_content)
                #print(1)

def make_batch_api_calls(file_path, start):
    i = 0
    for question in read_prefixes_in_batches(file_path):
        i += 1
        if i <= start:
            continue
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"{question}{suffix_prompt_1}"
                }
            ]
        )

        response = completion.choices[0].message.content
        #print(response)
        result = "'" + question + "'?" + "\t" + response
        yield result

        #print("'" + question + "'" + "\t" + response,  flush=True)
        #break

#make_batch_api_calls("CSE584_Final Project_Dataset.csv")

def validitytest(response):
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"{response}"
            }
        ]
    )

    vresponse = completion.choices[0].message.content

    return vresponse