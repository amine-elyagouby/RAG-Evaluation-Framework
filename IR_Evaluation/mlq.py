import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#dataset
with open("hotpot_dev_distractor_v1.json", 'r') as file:
    data = json.load(file)

#indexes
faiss_index = FAISS.load_local("./faiss_vector_store", embeddings, allow_dangerous_deserialization = True)

# multi queries
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model = 'llama3', temperature = 0.0)
# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> LineList:
        try:
            lines = [line.strip() for line in text.strip().split("\n")]
        except json.JSONDecodeError:
            raise ValueError("Failed to parse output as JSON")
        return lines
        




QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""I want you to decompose complex questions into their component simpler questions.

Instructions:
When given a complex question, break it down into a series of simpler questions that, when answered, will provide the necessary information to answer the original complex question.

Examples:

Example 1:
Complex Question: "Who is older, Real Madrid or FC Barcelona?"
Decomposition:

When was Real Madrid founded?
When was FC Barcelona founded?

Example 4:
Complex Question: "The Rand Paul presidential campaign, 2016 event was held at a hotel on what river?"
Decomposition:

Where was the Rand Paul presidential campaign, 2016 event held?
What river is near that location?

Example 2:
Complex Question: "Which is larger, the Amazon River or the Nile River?"
Decomposition:

What is the length of the Amazon River?
What is the length of the Nile River?

Example 5:
Complex Question: "George E. Blake was born in Yorkshire, England, formerly known as what?"
Decomposition:

Where was George E. Blake born?
What was Yorkshire, England formerly known as?

Example 3:
Complex Question: "Who has won more Nobel Prizes, Marie Curie or Albert Einstein?"
Decomposition:

How many Nobel Prizes did Marie Curie win?
How many Nobel Prizes did Albert Einstein win?

Example 6:
Complex Question: "What Netflix series, produced by Joe Swanberg, had an actress best known for her role as Vanessa on 'Atlanta'?"
Decomposition:

What Netflix series was produced by Joe Swanberg?
Who is the actress best known for her role as Vanessa on 'Atlanta'?
Which Netflix series produced by Joe Swanberg features this actress?
Your Task:
Please break down the following complex question into simpler component questions: "{question}".
responde only with these alternative questions separated by newlines, With no addtitionnel text?
""",
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=CustomOutputParser())
# Run
mlq_retriever = MultiQueryRetriever(
    retriever=faiss_index.as_retriever(search_kwargs={"k": 10}), llm_chain=llm_chain,parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output
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

    ir_results = mlq_retriever.invoke(query)
    ir_results = ir_results[:5]
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
with open('results_dev_mlq_1.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)