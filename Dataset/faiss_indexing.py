import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from tqdm import tqdm

print("Initializing SentenceTransformerEmbeddings")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading documents from 'corpus_chunks_2.json'")
with open('corpus_chunks_2.json', 'r') as file:
    documents = json.load(file)
print("len documents: ", len(documents))

print("Creating Document objects")
docs = [Document(page_content=doc[1]) for doc in documents]

db = 0
batch_size = 1000
print(f"Total number of documents: {len(docs)}")
with tqdm(total=len(docs), desc="Indexing documents",  position=0, leave=True) as pbar:
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if db:
            db.add_documents(batch)
        else:
            db = FAISS.from_documents(batch, embeddings)
        pbar.update(batch_size)

print("Saving FAISS vector store locally")
db.save_local("faiss_vector_store")

print("Process completed")
