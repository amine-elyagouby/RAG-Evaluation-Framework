import json
import pyterrier as pt
from tqdm import tqdm
if pt.started() == 0:
    pt.init()


def json_generator(filepath):
    print(f"Loading documents from '{filepath}'")
    with open(filepath, 'r') as f:
        documents = json.load(f)
    print(f"Total number of documents: {len(documents)}")
    with tqdm(total=len(documents), desc="bm25 indexing documents", position=0, leave=True) as pbar:
        for i, doc in enumerate(documents):
            chunk_doc = {}
            chunk_doc['docno'] = str(i)
            chunk_doc['text'] = doc[1]  # Assuming `chunk` should be replaced with `doc`
            yield chunk_doc
            pbar.update(1)
                
def index_corpus(corpus_path, index_path, generator = json_generator):
    meta = {
        'docno': 20,
        'text': 4000 
    }
    iter_indexer = pt.IterDictIndexer(index_path, meta=meta,overwrite=True, verbose=True)
    index_ref = iter_indexer.index(generator(corpus_path), fields=['text'])
    return index_ref

json_path = 'corpus_chunks.json'
index_path = './bm25_index'
index_ref = index_corpus(json_path, index_path, json_generator)
