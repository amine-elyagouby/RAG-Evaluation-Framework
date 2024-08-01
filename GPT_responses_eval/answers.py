import json
import ijson
from langchain_community.chat_models import ChatOllama
from tqdm import tqdm
from pathlib import Path

def get_C_P_sample(C, path = 'ir_scores/scores_bm25.json'):
    C_k = []
    P = []
    with open(path, 'rb') as file:
        for all_Ck in ijson.items(file, "all_Ck"):
            for index,_ in sample:
                C_k.append(all_Ck[index][:10])

    with open(path, 'rb') as file:
        for all_P in ijson.items(file, "all_P"):
            for index,_ in sample:
                P.append(all_P[index])
    return C_k, P

def get_C_P_2000(sample, path = 'ir_scores/scores_bm25.json'):
    C_2000 = []
    P = []
    with open(path, 'rb') as file:
        for all_Ck in ijson.items(file, "all_Ck"):
            for index,_ in sample:
                if len(all_Ck[index]) >= 10:
                    C_2000.append(all_Ck[index][19])
                else :
                    C_2000.append(all_Ck[index][-1])

    with open(path, 'rb') as file:
        for all_P in ijson.items(file, "all_P"):
            for index,_ in sample:
                P.append(all_P[index])            
    return C_2000, all_P

#rag
def generate_answer(query , documents=0, llm = ChatOllama(model='llama3', temperature=0.0)):
    if documents :
        prompt = f"""Given the documents below, your role is to answer the query using only these documents. Respond with a precise, one sentence explanation.
        
        Query:
        {query}
        
        Documents:
        {documents}
        
        Answer:"""
    else:
        prompt = f"""Responde with one sentence precise explanation to the given query.
        Query: {query}
        """
    result = llm.invoke(prompt)
    answer = result.content
    return answer

def generate_results(qa_data,C_k=0, filename = 'llm_only_answers_sample.json'):
    results = []
    if C_k :
        for (query, label_answer, supporting_facts), Ck in tqdm(zip(qa_data,C_k), total=len(qa_data), desc="level advance", position=0, leave=True):
            for C in Ck :
                answer = generate_answer(query, C)
                results.append({"query": query, 
                                "answer": answer, 
                               "label answer": label_answer,
                                "supporting_facts": supporting_facts
                               })
    
    else :
        for (query, label_answer, supporting_facts) in tqdm(qa_data, total=len(qa_data), desc="generating", position=0, leave=True):
            answer = generate_answer(query)
            results.append({"query": query, 
                            "answer": answer, 
                           "label answer": label_answer,
                            "supporting_facts": supporting_facts
                           })
            
    base = Path('rag_answers')
    jsonpath = base / filename
    base.mkdir(exist_ok=True)
    jsonpath.write_text(json.dumps(results))

def generate_results_2000(qa_data,C_k=0, filename = 'llm_only_answers_sample.json'):
    results = []
    if C_k :
        for (query, label_answer, supporting_facts), C in tqdm(zip(qa_data,C_k), total=len(qa_data), desc="level advance", position=0, leave=True):

            print(len(C))
            answer = generate_answer(query, C)
            
            results.append({"query": query, 
                            "answer": answer, 
                           "label answer": label_answer,
                            "supporting_facts": supporting_facts
                           })
    
    else :
        for (query, label_answer, supporting_facts) in tqdm(qa_data, total=len(qa_data), desc="generating", position=0, leave=True):
            answer = generate_answer(query)
            results.append({"query": query, 
                            "answer": answer, 
                           "label answer": label_answer,
                            "supporting_facts": supporting_facts
                           })
    base = Path('rag_answers')
    jsonpath = base / filename
    base.mkdir(exist_ok=True)
    jsonpath.write_text(json.dumps(results))

def get_qa_data(data, all_P):
    qa_data = []
    for example,facts in zip(data,all_P) : 
        qa_data.append((example['question'], example['answer'], facts))
    return qa_data
"""
with open("hotpot_dev_distractor_v1.json", 'r') as file:
    data = json.load(file)
"""   

with open("sample_100.json", 'r') as file:
    sample = json.load(file)

data = [s[1] for s in sample] 
print("Loading P...")
C_2000_bm25, all_P_bm25 = get_C_P_sample(sample, path='ir_scores/scores_bm25.json')
N = 2
print("Generating results for BM25...")
generate_results_2000(get_qa_data(data[:N], all_P_bm25[:N]), C_2000_bm25[:N], filename='llm_bm25_answers_sample_2000.json')
"""
print("Loading BM25 scores...")
C_1000_bm25, all_P_bm25 = get_C_P_sample(sample, path='ir_scores/scores_bm25.json')

print("Loading SIM scores...")
C_1000_sim, all_P_sim = get_C_P_sample(sample, path='ir_scores/scores_sim.json')

print("Loading MMR scores...")
C_1000_mmr, all_P_mmr = get_C_P_sample(sample, path='ir_scores/scores_mmr.json')

print("Loading REO scores...")
C_1000_reo, all_P_reo = get_C_P_sample(sample, path='ir_scores/scores_reo.json')

print("Loading MLQ scores...")
C_1000_mlq, all_P_mlq = get_C_P_1000(path='ir_scores/scores_mlq.json')

#N = 2
print("Generating results for MLQ...")
generate_results_1000(get_qa_data(data, all_P_mlq), C_1000_mlq, filename='llm_mlq_answers_sample.json')


print("Generating results for BM25...")
generate_results(get_qa_data(data[:N], all_P_bm25[:N]), C_1000_bm25[:N], filename='llm_bm25_answers_sample.json')

print("Generating results for SIM...")
generate_results(get_qa_data(data[:N], all_P_sim[:N]), C_1000_sim[:N], filename='llm_sim_answers_sample.json')

print("Generating results for MMR...")
generate_results(get_qa_data(data[:N], all_P_mmr[:N]), C_1000_mmr[:N], filename='llm_mmr_answers_sample.json')

print("Generating results for REO...")
generate_results(get_qa_data(data[:N], all_P_reo[:N]), C_1000_reo[:N], filename='llm_reo_answers_sample.json')

print("Generating results for MLQ...")
generate_results(get_qa_data(data[:N], all_P_mlq[:N]), C_1000_mlq[:N], filename='llm_mlq_answers_sample.json')


queries = []
for example in data : 
    queries.append(example['question'])

generate_results(queries)
"""