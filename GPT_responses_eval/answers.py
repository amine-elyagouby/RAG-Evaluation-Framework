import json
import ijson
from langchain_community.chat_models import ChatOllama
from tqdm import tqdm
from pathlib import Path


def (sample, path = 'ir_scores/scores_bm25.json'):
    C_1000 = []
    P = []
    with open(path, 'rb') as file:
        for all_Ck in ijson.items(file, "all_Ck"):
            for index,_ in sample:
                C_1000.append(all_Ck[index][:10])

    with open(path, 'rb') as file:
        for all_P in ijson.items(file, "all_P"):
            for index,_ in sample:
                P.append(all_P[index])            
    return C_1000, all_P

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

def generate_results(qa_data,C_k=0, filename = 'llm_only_answers.json'):
    results = []
    if C_k :
        for (query, label_answer, supporting_facts), Ck in tqdm(zip(qa_data,C_k), total=len(qa_data), desc="generating", position=0, leave=True):
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

def get_qa_data(data, all_P):
    qa_data = []
    for example, facts in zip(data,all_P) : 
        qa_data.append((example['question'], example['answer'], facts))
    return qa_data

with open("../data_corpus/hotpot_dev_distractor_v1.json", 'r') as file:
    data = json.load(file)
   
with open("sample_100.json", 'r') as file:
    sample = json.load(file)


print("Loading contexts...")
print("Loading BM25 contexts...")
C_1000_bm25, all_P_bm25 = get_C_P_sample(sample, path='ir_scores/scores_bm25.json')

print("Loading SIM contexts...")
C_1000_sim, all_P_sim = get_C_P_sample(sample, path='ir_scores/scores_sim.json')

print("Loading MMR contexts...")
C_1000_mmr, all_P_mmr = get_C_P_sample(sample, path='ir_scores/scores_mmr.json')

print("Loading REO contexts...")
C_1000_reo, all_P_reo = get_C_P_sample(sample, path='ir_scores/scores_reo.json')

print("Loading MLQ contexts...")
C_1000_mlq, all_P_mlq = (path='ir_scores/scores_mlq.json')


print("Generating results for LLMOnly...")
generate_results(get_qa_data(data, all_P_mlq), 0, filename='llm_only_answers.json')

print("Generating results for BM25...")
generate_results(get_qa_data(data, all_P_bm25), C_1000_bm25, filename='llm_bm25_answers.json')

print("Generating results for SIM...")
generate_results(get_qa_data(data, all_P_sim), C_1000_sim, filename='llm_sim_answers.json')

print("Generating results for MMR...")
generate_results(get_qa_data(data, all_P_mmr), C_1000_mmr, filename='llm_mmr_answers.json')

print("Generating results for REO...")
generate_results(get_qa_data(data, all_P_reo), C_1000_reo, filename='llm_reo_answers.json')

print("Generating results for MLQ...")
generate_results(get_qa_data(data, all_P_mlq), C_1000_mlq, filename='llm_mlq_answers.json')




