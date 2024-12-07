import openaitextcompletion
import groqcompletion
import geminitextcompletion
import time
prefix = "I asked an invalid question to an LLM, meaning the question was attempting to fool the LLM. The question and the LLM's response are: '"
suffix = "'. Now, can you tell me, from the LLM's response, whether that LLM recognized that the question was invalid? Just answer 'yes' or 'no'."

gemini_judgement = 0
groq_judgement = 0
i = 40
for response in openaitextcompletion.make_batch_api_calls("CSE584_Final Project_Dataset.csv", 40):
    validityprompt = prefix + response + suffix
    gemini_judge = geminitextcompletion.validitytest(validityprompt)
    groq_judge = groqcompletion.validitytest(validityprompt)

    if "yes" in gemini_judge.lower():
        gemini_judgement += 1

    if "yes" in groq_judge.lower():
        groq_judgement += 1

    i += 1

    print(i,  flush=True)
    print(validityprompt,  flush=True)
    print(f"GEMINI JUDGE: {gemini_judge}",  flush=True)
    print(f"GROQ JUDGE: {groq_judge}",  flush=True)

    #time.sleep(30)

print(f"GEMINI can tell {gemini_judgement} reponses of openai as understandable out of {i}")
print(f"GROQ can tell {groq_judgement} reponses of openai as understandable out of {i}")
