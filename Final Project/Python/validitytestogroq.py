import openaitextcompletion
import groqcompletion
import geminitextcompletion
import time
prefix = "I asked an invalid question to an LLM, meaning the question was attempting to fool the LLM. The question and the LLM's response are: '"
suffix = "'. Now, can you tell me, from the LLM's response, whether that LLM recognized that the question was invalid? Just answer 'yes' or 'no'."

gemini_judgement = 0
openai_judgement = 0
i = 0
for response in groqcompletion.make_batch_api_calls("CSE584_Final Project_Dataset.csv", 0):
    validityprompt = prefix + response + suffix
    gemini_judge = geminitextcompletion.validitytest(validityprompt)
    openai_judge = openaitextcompletion.validitytest(validityprompt)

    if "yes" in gemini_judge.lower():
        gemini_judgement += 1

    if "yes" in openai_judge.lower():
        openai_judgement += 1

    i += 1

    print(i,  flush=True)
    print(validityprompt,  flush=True)
    print(f"GEMINI JUDGE: {gemini_judge}",  flush=True)
    print(f"OPENAI JUDGE: {openai_judge}",  flush=True)

    #time.sleep(20)

print(f"GEMINI can tell {gemini_judgement} reponses of GROQ as understandable out of {i}")
print(f"OPENAI can tell {openai_judgement} reponses of GROQ as understandable out of {i}")
