import json
from tqdm import tqdm
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

with open("hotpot_dev_distractor_v1.json", 'r') as file:
    data = json.load(file)
"""
with open('results_dev_bm25.json', 'r') as file:
    results_bm25 = json.load(file)

with open('results_dev_sim_1.json', 'r') as file:
    results_sim = json.load(file)

"""
with open('results_dev_reo_1.json', 'r') as file:
    results_reo = json.load(file)
    
with open('results_dev_mlq_1.json', 'r') as file:
    results_mlq = json.load(file)


with open('results_dev_mmr_1.json', 'r') as file:
    results_mmr = json.load(file)


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

def get_score(P, Ck):
    scores = []
    for C in Ck:
        score = average_subsentence_score(P, C)
        scores.append(score)
    return scores
def get_score(P, Ck):
    scores = []
    for C in Ck:
        score = average_subsentence_score(P, C)
        scores.append(score)
    return scores
def get_scores_P_Ck(results):
    all_P = []
    all_Ck = []
    all_scores = []
    lens = []
    for res in tqdm(results, total=len(results), desc="Processing", position=0, leave=True):
        P = []
        Ck = []
        for f in res['supporting_facts']:
            fact = f[0]
            P.append(fact)
        text = ' '.join(res['retrieved_chunks'])
        token_ids = tokenizer.encode(text)
        N = len(token_ids)
        list_N = list(range(100, N + 100, 100))

        Ck = []
        for N in list_N:
            first_N_token_ids = token_ids[1:N]
            first_N_tokens_text = tokenizer.decode(first_N_token_ids)
            Ck.append(first_N_tokens_text)

        scores = get_score(P, Ck)
        all_scores.append(scores)
        all_Ck.append(Ck)
        all_P.append(P)
        lens.append(N)
    return all_scores, all_Ck, all_P, lens



from pathlib import Path
def save_scores_to_json(results, filename, dir):
    all_scores, all_Ck, all_P, lens = get_scores_P_Ck(results)
    output = {
        "all_scores": all_scores,
        "all_Ck": all_Ck,
        "all_P" : all_P,
        "lens": lens
    }
    base = Path(dir)
    jsonpath = base / filename
    base.mkdir(exist_ok=True)
    jsonpath.write_text(json.dumps(output))

# Calculate and save scores for each result set
save_scores_to_json(results_mmr, 'scores_mmr.json','ir_scores')
save_scores_to_json(results_mlq, 'scores_mlq.json', 'ir_scores')
save_scores_to_json(results_reo, 'scores_reo.json', 'ir_scores')
"""
save_scores_to_json(results_sim, 'scores_sim.json', 'ir_scores')
save_scores_to_json(results_bm25, 'scores_bm25.json', 'ir_scores')
"""