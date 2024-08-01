#bm25
import json

import pyterrier as pt
if not pt.started():
    pt.init()
import re

bm25_index = pt.IndexFactory.of('./bm25_index')

#eval metric
def longest_subsentence_in_text(sentence, text):
    len_sentence = len(sentence)
    for l in range(len_sentence, 0, -1):
        for start in range(len_sentence - l + 1):
            subsentence = sentence[start:start + l]
            if subsentence in text:
                return l
    return 0

def average_subsentence_score(sentences, text):
    scores = []
    for sentence in sentences:
        l = longest_subsentence_in_text(sentence, text)
        len_sentence = len(sentence)
        score = l / len_sentence
        scores.append(score)
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score, scores

class BM25_Retriever:
    def __init__(self, index):
        self.index = index
        self.retriever = pt.BatchRetrieve(self.index, wmodel='BM25', metadata=["docno", "text"]) % 5 >> pt.text.get_text(self.index, "text")

    def get_relevant_documents(self, query):
        if query.endswith('?'):
            query = query[:-1]
        if query.endswith(' '):
            query = query[:-2]
        query = query.replace("'", "")
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        results = self.retriever.search(query)
        documents = [Document(page_content=row["text"]) for _, row in results.iterrows()]
        #documents = [row["text"] for _, row in results.iterrows()]
        return documents

bm25 = BM25_Retriever(bm25_index)

#dataset
with open("hotpot_dev_distractor_v1.json", 'r') as file:
    data = json.load(file)


# sim ret
#eval on dataset storethe res in json files
from tqdm import tqdm
results = []
for d in tqdm(data, total=len(data), desc="Processing", position=0, leave=True):
    result_dict = {}
    query = d['question']
    answer = d['answer']
    supporting_facts = []
    type = ['type']
    level = ['level']
    
    facts = []
    for f in d['supporting_facts']:
        title = f[0]
        index = f[1]
        
        for c in d['context']:  
            if title == c[0]:
                facts.append(c[1][index])

    bm25_passages = bm25.get_relevant_documents(query)
    text = " ".join(bm25_passages)
    avg_score, scores = average_subsentence_score(facts, text)
    #rag_answer = generate_answer(query , documents=text, RAG = True)
    rag_answer = 0

    for i,fact in enumerate(facts):
        supporting_facts.append((fact,scores[i]))

    result_dict['question'] = query
    result_dict['answer'] = answer
    result_dict['supporting_facts'] = supporting_facts
    result_dict['type'] = type
    result_dict['level'] = level
    result_dict['avg_score'] = avg_score
    result_dict['rag_answer'] = rag_answer
    result_dict['retrieved_chunks'] = bm25_passages

    results.append(result_dict)

# Save results to a JSON file
with open('results_dev_bm25_1.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)