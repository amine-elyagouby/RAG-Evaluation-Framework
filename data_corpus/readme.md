## Dataset and Corpus

1. **Download the Dataset and Corpus:**

   Download the dataset and corpus files from [Kaggle](https://www.kaggle.com/datasets/amineelyagouby/rag-evaluation-data).

   - **Dataset file:** `hotpot_dev_distractor_v1.json`
   - **Corpus file:** `corpus_100k.json`

2. **Corpus Chunking:**

   Run `chunking.py` to create a corpus of chunks (`corpus_chunks.json`) from the `corpus_100k.json`. Alternatively, you can directly download the `corpus_chunks.json` file from the Kaggle link above.

   ```sh
   python chunking.py
   ```

2. **Indexing:**

* BM25 Indexing

   Create an index for BM25 using PyTerrier by running the following script:

   ```sh
   python bm25_indexing.py
   ```

* Faiss Indexing

   Create a Faiss vector store index using embeddings for the other 4 retrievals based on similarity between embeddings. Run the following script:

   ```sh
   python faiss_indexing.py
   ```

## Files Description

- `hotpot_dev_distractor_v1.json`: HotpotQA Dataset for evaluation.
- `corpus_100k.json`: Original corpus of 100k documents (Wikipedia pages) including relevent documents for all the dataset queries.
- `corpus_chunks.json`: Chunks of the original corpus created using `chunking.py`.
- `chunking.py`: Script to create chunks from the original corpus.
- `bm25_indexing.py`: Script to create BM25 index using PyTerrier.
- `faiss_indexing.py`: Script to create Faiss vector store index using embeddings.


## Usage

1. **Download the required files** from the provided Kaggle link.
2. **Run the chunking script** to preprocess the corpus (if not using the pre-chunked file).
3. **Index the chunks** using BM25 and Faiss by running the respective scripts.
