import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Function to split text into sentences
def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    merged_sentences = []
    buffer = ""
    
    for sentence in sentences:
        if len(buffer) <= 200:
            sentence = buffer + " " + sentence
            buffer = ""
        
        if len(sentence) < 200:
            buffer = sentence
        else:
            merged_sentences.append(sentence)
    
    if len(buffer) > 0:
        merged_sentences.append(buffer)
    return merged_sentences,buffer

# Function to combine sentences with a buffer size
def combine_sentences(sentences, buffer_size=1):
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = ' '.join(sentences[max(i - buffer_size, 0): min(i + 1 + buffer_size, len(sentences))])
        combined_sentences.append({'sentence': sentences[i], 'combined_sentence': combined_sentence, 'index': i})
    return combined_sentences

# Function to calculate cosine distances between sentence embeddings
def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance
    return distances, sentences

# Function to chunk text based on calculated distances
def chunk_text_based_on_distances(sentences, distances, percentile_threshold=95):
    breakpoint_distance_threshold = np.percentile(distances, percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    chunks = []
    start_index = 0
    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)
        start_index = index + 1

    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)
    
    return chunks

# Main function to process a list of texts and perform semantic chunking
from tqdm import tqdm

def semantic_chunking(texts, buffer_size=1, percentile_threshold=95, embeddings=embeddings):
    all_chunks = []
    oaiembeds = embeddings

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(texts), desc="Processing texts",  position=0, leave=True)

    for idx, text in enumerate(texts):
        title = text['title']
        sentences, _ = split_text_into_sentences(text['text'])
        if len(sentences) == 1:
            chunks = sentences
        else:
            combined_sentences = combine_sentences(sentences, buffer_size)
            embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in combined_sentences])

            for i, sentence in enumerate(combined_sentences):
                sentence['combined_sentence_embedding'] = embeddings[i]

            distances, combined_sentences = calculate_cosine_distances(combined_sentences)
            chunks = chunk_text_based_on_distances(combined_sentences, distances, percentile_threshold)
        
        chunks_a = []
        for chunk in chunks:
            chunks_a.append([title, chunk])
        all_chunks.extend(chunks_a)

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()
    print('dump into a json file')
    with open('corpus_chunks_2.json', 'w') as json_file:
        json.dump(all_chunks, json_file, indent=4)
    print('end')

    return all_chunks

with open('corpus_100k_1.json', 'r') as file:
    documents = json.load(file)

chunks = semantic_chunking(documents, buffer_size=3, percentile_threshold=80)