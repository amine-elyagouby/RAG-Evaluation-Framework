import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


#dataset
with open("hotpot_dev_distractor_v1.json", 'r') as file:
    data = json.load(file)

faiss_index = FAISS.load_local("./faiss_vector_store", embeddings, allow_dangerous_deserialization = True)

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

#eval on dataset storethe res in json files
from tqdm import tqdm
mmr_retriever = faiss_index.as_retriever(search_type="mmr", search_kwargs={"k": 10})
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

    ir_results = mmr_retriever.get_relevant_documents(query)
    retrieved_chunks = [doc.page_content for doc in ir_results]
    text = " ".join(retrieved_chunks)
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
    result_dict['retrieved_chunks'] = retrieved_chunks


    results.append(result_dict)

# Save results to a JSON file
with open('results_dev_mmr_1.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)