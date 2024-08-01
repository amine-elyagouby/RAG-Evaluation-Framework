import json
from langchain_community.chat_models import ChatOllama
from tqdm import tqdm
import os


from openai import OpenAI
client = OpenAI(
    # Add your GPT API Key here
    api_key='',
)

with open("rag_answers/llm_bm25_answers.json", 'r') as file:
    answers_bm25 = json.load(file)
with open("rag_answers/llm_sim_answers.json", 'r') as file:
    answers_sim = json.load(file)
with open("rag_answers/llm_mmr_answers.json", 'r') as file:
    answers_mmr = json.load(file)
with open("rag_answers/llm_mlq_answers.json", 'r') as file:
    answers_mlq = json.load(file)
with open("rag_answers/llm_reo_answers.json", 'r') as file:
    answers_reo = json.load(file)
with open("rag_answers/llm_only_answers.json", 'r') as file:
    answers_llm = json.load(file)

prompt_template = """Task: Assess the candidate's answers based on the true answer, references, and scoring criteria, and return only the scores separated by commas.
Question: "{question}" 

True Answer: "{true_answer}"

References:
{formatted_facts}

Candidate's Answers:
{formatted_answers}
Scoring Criteria:

If the candidate said that they cannot find enough information in documents to answer the question, the score is 1.
If the answer is partially correct but in details you found statements that are incorrect, the score is 2.
If the answer is partially correct but you found that it don't completely answer the question, the score is 3.
If the answer is fully incorrect, the score is 4.
If the answer is fully correct, the score is 5.
"""

all_scores = []
output_file = "gpt_scores.txt"

# Ensure the file exists or create it if it doesn't
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        f.write('')  # create an empty file
        
all_scores = []
for i in tqdm(range(len(answers_bm25)), total=len(answers_bm25), desc="gpt evaluation", position=0, leave=True):
    answers = []
    answers.append(answers_llm[i]['answer'])
    answers.append(answers_bm25[i]['answer'])
    answers.append(answers_sim[i]['answer'])
    answers.append(answers_mmr[i]['answer'])
    answers.append(answers_reo[i]['answer'])
    answers.append(answers_mlq[i]['answer'])
    formatted_answers = '\n'.join([f'candidate answer {idx+1}: "{answer}"' for idx, answer in enumerate(answers)])
    formatted_facts = '\n'.join([f'"{fact}"' for fact in answers_bm25[i]['supporting_facts']])
    prompt = prompt_template.format(question = answers_bm25[i]['query'], true_answer = answers_bm25[i]['label answer'], formatted_facts = formatted_facts,
            formatted_answers = formatted_answers)
    
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
        ],
        model="gpt-4o",
        temperature=0,
    )
    response_text = chat_completion.choices[0].message.content.strip()
    scores = list(map(int, response_text.split(',')))
    all_scores.append(scores)
    
        # Append scores to file
    with open(output_file, 'a') as f:
        f.write(','.join(map(str, scores)) + '\n')

"""
with open("gpt_scores.json", 'w') as json_file:
     json.dump(all_scores, json_file, indent=4)
"""
