import openaitextcompletion
import groqcompletion
import geminitextcompletion
import time
prefix = "I asked an invalid question to an LLM, meaning the question was attempting to fool the LLM. The question and the LLM's response are: '"
suffix = "'. Now, can you tell me, from the LLM's response, whether that LLM recognized that the question was invalid? Just answer 'yes' or 'no'."

openai_judgement = 0
groq_judgement = 0
i = 72
for response in geminitextcompletion.make_batch_api_calls("CSE584_Final Project_Dataset.csv", 72):
    validityprompt = prefix + response + suffix
    openai_judge = openaitextcompletion.validitytest(validityprompt)
    groq_judge = groqcompletion.validitytest(validityprompt)

    if "yes" in openai_judge.lower():
        openai_judgement += 1

    if "yes" in groq_judge.lower():
        groq_judgement += 1

    i += 1

    print(i,  flush=True)
    print(validityprompt,  flush=True)
    print(f"OPENAI JUDGE: {openai_judge}",  flush=True)
    print(f"GROQ JUDGE: {groq_judge}",  flush=True)

    #time.sleep(20)

print(f"OPENAI can tell {openai_judgement} reponses of gemini as understandable out of {i}")
print(f"GROQ can tell {groq_judgement} reponses of gemini as understandable out of {i}")
