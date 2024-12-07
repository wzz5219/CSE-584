import google.generativeai as genai
import os
import sys
import time
import pandas as pd
import csv

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
suffix_prompt = ". Please make sure the question is valid beforehand."

model = genai.GenerativeModel("gemini-1.5-pro")

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
        if i <= start :
            continue
        response = model.generate_content(
            f"{question}")

        result = "'" + question + "'?" + "\t" + response.text
        yield result
        #print("'" + question + "'" + "\t" + response.text,  flush=True)
        #break
            
#make_batch_api_calls("CSE584_Final Project_Dataset.csv")

def validitytest(response):
    vresponse = model.generate_content(
            f"{response}")

    return vresponse.text