#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SBATCH directives (SLURM job configuration)
# These lines are comments and will be ignored by Python

#SBATCH --job-name=5M-pub-test-head-raw    # Name of the job
#SBATCH --output=head-raw.out    # Standard output log file
#SBATCH --error=head-raw.err     # Standard error log file
#SBATCH --partition=bigmem                   # Request the bigmem partition (CPU)
#SBATCH --ntasks=1                           # Number of tasks (1 process)
#SBATCH --cpus-per-task=30                   # Number of CPU cores requested per task
#SBATCH --mem=200G                            # Amount of RAM requested
#SBATCH --time=2-00:00:00                      # Runtime (D-HH:MM:SS format)
#SBATCH --mail-type=BEGIN,END,FAIL           # Send email notifications on job completion or failure
#SBATCH --mail-user=rgoli@student.unimelb.edu.au   # Your email address



import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
import time
import numpy as np
from scipy.special import softmax
import psutil
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from scipy.special import softmax
import pickle

import psutil

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


sys.stdout = open("head-raw.txt", "w")
sys.stderr = sys.stdout  # Redirect stderr to the same file as stdout

# Get Data - HEAD Queries
# queries File paths
head_queries_train_file = '../queries/topics.head.train.tsv'
head_queries_val_file = '../queries/topics.head.val.tsv'
head_queries_test_file = '../queries/topics.head.test.tsv'

# Read the TSV files
head_queries_train_df = pd.read_csv(head_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
head_queries_val_df = pd.read_csv(head_queries_val_file, delimiter='\t', header=None, names=['qid', 'qtext'])
head_queries_test_df = pd.read_csv(head_queries_test_file, delimiter='\t', header=None, names=['qid', 'qtext'])

# Display the combined DataFrame
print("head_queries_train_df.shape[0]",head_queries_train_df.shape[0])


# Get Data - TORSO Queries
# queries File paths
torso_queries_train_file = '../queries/topics.torso.train.tsv'
torso_queries_val_file = '../queries/topics.torso.val.tsv'
torso_queries_test_file = '../queries/topics.torso.test.tsv'

# Read the TSV files
torso_queries_train_df = pd.read_csv(torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
torso_queries_val_df = pd.read_csv(torso_queries_val_file, delimiter='\t', header=None, names=['qid', 'qtext'])
torso_queries_test_df = pd.read_csv(torso_queries_test_file, delimiter='\t', header=None, names=['qid', 'qtext'])


# Display the combined DataFrame
print("torso_queries_train_df.shape[0]",torso_queries_train_df.shape[0])


# Get Data - TAIL Queries
# queries File paths
tail_queries_train_file = '../queries/topics.tail.train.tsv'
tail_queries_val_file = '../queries/topics.tail.val.tsv'
tail_queries_test_file = '../queries/topics.tail.test.tsv'

# Read the TSV files
tail_queries_train_df = pd.read_csv(tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
tail_queries_val_df = pd.read_csv(tail_queries_val_file, delimiter='\t', header=None, names=['qid', 'qtext'])
tail_queries_test_df = pd.read_csv(tail_queries_test_file, delimiter='\t', header=None, names=['qid', 'qtext'])


# Display the combined DataFrame
print("tail_queries_train_df.shape[0]",tail_queries_train_df.shape[0])

# Get Data Query Relevant Score
# queries File paths
dctr_head_queries_train_file = '../qrels/qrels.dctr.head.train.tsv'
dctr_head_queries_val_file = '../qrels/qrels.dctr.head.val.tsv'
dctr_head_queries_test_file = '../qrels/qrels.dctr.head.test.tsv'

raw_head_queries_train_file = '../qrels/qrels.raw.head.train.tsv'
raw_head_queries_val_file = '../qrels/qrels.raw.head.val.tsv'
raw_head_queries_test_file = '../qrels/qrels.raw.head.test.tsv'

raw_torso_queries_train_file = '../qrels/qrels.raw.torso.train.tsv'
raw_torso_queries_val_file = '../qrels/qrels.raw.torso.val.tsv'
raw_torso_queries_test_file = '../qrels/qrels.raw.torso.test.tsv'

raw_tail_queries_train_file = '../qrels/qrels.raw.tail.train.tsv'
raw_tail_queries_val_file = '../qrels/qrels.raw.tail.val.tsv'
raw_tail_queries_test_file = '../qrels/qrels.raw.tail.test.tsv'

# Read the TSV files
dctr_head_queries_train_df = pd.read_csv(dctr_head_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
dctr_head_queries_val_df = pd.read_csv(dctr_head_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
dctr_head_queries_test_df = pd.read_csv(dctr_head_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])

raw_head_queries_train_df = pd.read_csv(raw_head_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
raw_head_queries_val_df = pd.read_csv(raw_head_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
raw_head_queries_test_df = pd.read_csv(raw_head_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])

raw_torso_queries_train_df = pd.read_csv(raw_torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
raw_torso_queries_val_df = pd.read_csv(raw_torso_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
raw_torso_queries_test_df = pd.read_csv(raw_torso_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])

raw_tail_queries_train_df = pd.read_csv(raw_tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
raw_tail_queries_val_df = pd.read_csv(raw_tail_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
raw_tail_queries_test_df = pd.read_csv(raw_tail_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])


dctr_head_queries_train_df = dctr_head_queries_train_df.drop(dctr_head_queries_train_df.columns[1], axis=1)
dctr_head_queries_val_df = dctr_head_queries_val_df.drop(dctr_head_queries_val_df.columns[1], axis=1)
dctr_head_queries_test_df = dctr_head_queries_test_df.drop(dctr_head_queries_test_df.columns[1], axis=1)

raw_head_queries_train_df = raw_head_queries_train_df.drop(raw_head_queries_train_df.columns[1], axis=1)
raw_head_queries_val_df = raw_head_queries_val_df.drop(raw_head_queries_val_df.columns[1], axis=1)
raw_head_queries_test_df = raw_head_queries_test_df.drop(raw_head_queries_test_df.columns[1], axis=1)

raw_torso_queries_train_df = raw_torso_queries_train_df.drop(raw_torso_queries_train_df.columns[1], axis=1)
raw_torso_queries_val_df = raw_torso_queries_val_df.drop(raw_torso_queries_val_df.columns[1], axis=1)
raw_torso_queries_test_df = raw_torso_queries_test_df.drop(raw_torso_queries_test_df.columns[1], axis=1)

raw_tail_queries_train_df = raw_tail_queries_train_df.drop(raw_tail_queries_train_df.columns[1], axis=1)
raw_tail_queries_val_df = raw_tail_queries_val_df.drop(raw_tail_queries_val_df.columns[1], axis=1)
raw_tail_queries_test_df =raw_tail_queries_test_df.drop(raw_tail_queries_test_df.columns[1], axis=1)


# Combine the train for head+ torso+ tail
qrel_train_df = pd.concat([raw_head_queries_train_df, raw_torso_queries_train_df, raw_tail_queries_train_df], ignore_index=True)
qrel_val_df = pd.concat([dctr_head_queries_val_df, raw_torso_queries_val_df, raw_tail_queries_val_df], ignore_index=True)

# Display the combined DataFrame
print("dctr_head_queries_train_df.shape[0]", dctr_head_queries_train_df.shape[0])


# ### Get Data - Documents
documentst_file = '../documents/docs.tsv'

# Read the TSV files
documentst_df = pd.read_csv(documentst_file, delimiter='\t', header=None, names=['docid', 'doctext'])
print("documentst_df.shape[0]",documentst_df.shape[0])


def split_title_abstract(text):
    # Check for the presence of markers
    if '<eot>' in text:
        title, abstract = text.split('<eot>', 1)
    elif 'BACKGROUND :' in text:
        title, abstract = text.split('BACKGROUND :', 1)
    else:
        # If neither marker is present, use the entire text as the title and leave abstract empty
        return text, ''
    
    # Clean up the title and abstract
    title = title.strip()
    abstract = abstract.strip()
    
    return title, abstract

# Apply the function to the `doctext` column
documentst_df[['title', 'abstract']] = documentst_df['doctext'].apply(split_title_abstract).apply(pd.Series)

print("documentst_df.shape[0]", documentst_df.shape[0])


# Merge head, torso, and tail datasets for training, validation, and testing
train_queries_df = pd.concat([head_queries_train_df, torso_queries_train_df, tail_queries_train_df])
val_queries_df = pd.concat([head_queries_val_df, torso_queries_val_df, tail_queries_val_df])
test_queries_df = pd.concat([head_queries_test_df, torso_queries_test_df, tail_queries_test_df])

print("train_queries_df.shape[0]", train_queries_df.shape[0])
print("val_queries_df.shape[0]",val_queries_df.shape[0])
print("test_queries_df.shape[0]", test_queries_df.shape[0])


# loading the trained model 
doc_encoder_path = r"../../trained_model_initial_PubMed_all/checkpoint-40000/doc_encoder"
query_encoder_path = r"../../trained_model_initial_PubMed_all/checkpoint-40000/query_encoder"

# Load the document encoder
doc_tokenizer = AutoTokenizer.from_pretrained(r"../../trained_model_initial_PubMed_all/final_tokenizer")
doc_encoder = AutoModel.from_pretrained(doc_encoder_path)

# Load the query encoder
query_tokenizer = AutoTokenizer.from_pretrained(r"../../trained_model_initial_PubMed_all/final_tokenizer")
query_encoder = AutoModel.from_pretrained(query_encoder_path)


def get_embeddings(texts, model, tokenizer, max_length=256):
    # Tokenize input text
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    
    # Pass the inputs through the model to get the embeddings
    with torch.no_grad():  # Turn off gradients for inference
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract embeddings from the output (e.g., using the hidden states of the [CLS] token)
    embeddings = outputs.hidden_states[-1][:, 0, :]  # Use the [CLS] token embeddings
    return embeddings


def log_memory_usage(log_times):
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 3)  # Memory usage in GB
    log_times += 1  # Increment the log count
    print(f"Log {log_times}: Memory usage: {memory_usage:.2f} GB")
    return log_times


def get_embeddings_batch(texts, model, tokenizer, batch_size):

    log_times = 0  # Counter for how many times memory is logged
    all_embeddings = []
    # start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Simulate embedding extraction
        batch_embeddings = get_embeddings(batch, model, tokenizer)
        all_embeddings.append(torch.tensor(batch_embeddings))

        # Log memory usage after every 100 batches
        if (i // batch_size) % 100 == 0 and i > 0: 
            log_times = log_memory_usage(log_times)

    return torch.cat(all_embeddings)



# get embeddings of the queries in train set 
# start_time = time.time()
# query_texts = train_queries_df['qtext'].tolist()
# print("the number of query in list :", len(query_texts))
# query_embeddings = get_embeddings_batch(query_texts, query_encoder, query_tokenizer, batch_size=100)
# print("Number of vectors (rows):", query_embeddings.size(0))
# print("Size of each vector (columns):", query_embeddings.size(1))
# end_time = time.time()
# duration = end_time - start_time
# print(f"Duration: {duration} seconds")

# # save the query embeddings
# torch.save(query_embeddings, 'query_embeddings_5M_pubmed.pt')

# load the model
query_embeddings = torch.load('query_embeddings_5M_pubmed.pt')

print("Number of vectors (rows):", query_embeddings.size(0))
print("Size of each vector (columns):", query_embeddings.size(1))


# get embeddings of documents 
# start_time = time.time()
# log_times = 0  # Counter for how many times memory is logged
# document_texts = documentst_df['doctext'].tolist()
# batch_size = 100
# document_embeddings = get_embeddings_batch(document_texts, doc_encoder, doc_tokenizer, batch_size)
    
# # End of the processing, log the final memory usage and time
# log_times = log_memory_usage(log_times)
# end_time = time.time()
# duration = end_time - start_time
# print(f"Total time: {duration:.2f} seconds")
# torch.save(document_embeddings, 'document_embeddings_5M_pubmed.pt')

document_embeddings = torch.load('document_embeddings_5M_pubmed.pt')

print("Number of vectors (rows):", document_embeddings.size(0))
print("Size of each vector (columns):", document_embeddings.size(1))


def compute_inner_products(query_embeddings_batch,embedding_batch):
    inner_products_batch = np.dot(query_embeddings_batch, embedding_batch.T)
    return inner_products_batch


def brute_force_mips_batch(query_embeddings_batch, embeddings, k, embedding_batch_size):

    # Initialize global top scores and indices for all queries
    global_top_indices = np.zeros((query_embeddings_batch.shape[0], k), dtype=int)
    global_top_scores = np.full((query_embeddings_batch.shape[0], k), -np.inf)  
    
    # Process embeddings in smaller batches
    for i in range(0, embeddings.shape[0], embedding_batch_size):
        # Select the current embedding batch
        embedding_batch = embeddings[i:i + embedding_batch_size]
        inner_products_batch= compute_inner_products(query_embeddings_batch,embedding_batch)
        
        # For each query in the batch, track the top-k scores and indices across all embedding batches
        for query_idx, inner_products in enumerate(inner_products_batch):

            top_indices_in_batch = np.argsort(inner_products)[-k:][::-1]
            top_scores_in_batch = inner_products[top_indices_in_batch]

            # Combine current top-k with global top-k results
            combined_scores = np.concatenate([global_top_scores[query_idx], top_scores_in_batch])
            combined_indices = np.concatenate([global_top_indices[query_idx], top_indices_in_batch + i])  # Adjust indices based on batch

            # Get the overall top-k from the combined results
            top_k_combined = np.argsort(combined_scores)[-k:][::-1]

            # Update global top-k scores and indices
            global_top_scores[query_idx] = combined_scores[top_k_combined]
            global_top_indices[query_idx] = combined_indices[top_k_combined]

    
    return global_top_indices, global_top_scores


def perform_mips_batch(new_queries, query_embeddings, document_embeddings, k, lambda_param=0.5, query_batch_size=100, embedding_batch_size=100):

    # Encode new queries in batches
    all_query_embeddings = get_embeddings_batch(new_queries, query_encoder, query_tokenizer, query_batch_size)

    print("new query embedding created!")
    print("Number of vectors (rows):", all_query_embeddings.size(0))
    print("Size of each vector (columns):", all_query_embeddings.size(1))


    ### Add Normaliation
    # norms = torch.norm(all_query_embeddings, dim=1, keepdim=True)
    # # Divide each vector by its norm to normalize
    # all_query_embeddings = all_query_embeddings / norms
    # # Check the result
    # print("Shape of normalized embeddings:", all_query_embeddings.shape)
    # print("Norms after normalization (should be close to 1):", torch.norm(all_query_embeddings, dim=1))

    log_times=0
    # Initialize lists to hold global top-k results
    all_top_queries_indices = []
    all_top_queries_similarities = []
    all_top_documents_indices = []
    all_top_documents_similarities = []
    
    # Process queries and documents in batches
    for i in range(0, len(all_query_embeddings), query_batch_size):

        # Get the current batch of queries
        query_batch = all_query_embeddings[i:i + query_batch_size]
        
        # Perform MIPS for the current query batch with global top-k tracking
        top_queries_indices, top_queries_similarities = brute_force_mips_batch(
            query_batch, query_embeddings, k=k, embedding_batch_size=embedding_batch_size
        )

        all_top_queries_indices.extend(top_queries_indices)
        all_top_queries_similarities.extend(top_queries_similarities)
        
        # Perform MIPS for documents in the batch with global top-k tracking
        top_documents_indices, top_documents_similarities = brute_force_mips_batch(
            query_batch, document_embeddings, k=k, embedding_batch_size=embedding_batch_size
        )
        
        all_top_documents_indices.extend(top_documents_indices)
        all_top_documents_similarities.extend(top_documents_similarities)

        # Log memory usage after every 5 batches
        if (i // query_batch_size) % 1 == 0 and i > 0:  # After every 2 batches
            log_times = log_memory_usage(log_times)

    # Normalize the similarities (optional)
    normalized_query_similarities = softmax(all_top_queries_similarities)
    normalized_document_similarities = softmax(all_top_documents_similarities)

    # normalized_query_similarities = all_top_queries_similarities
    # normalized_document_similarities = all_top_documents_similarities
    
    return all_top_queries_indices, all_top_documents_indices, normalized_query_similarities, normalized_document_similarities
    


### head queries --> test
new_query_text = head_queries_test_df['qtext'].tolist()
new_query_ids = head_queries_test_df['qid'].tolist()
print("head_queries_test_df.shape[0]", head_queries_test_df.shape[0])


# # Perform MIPS with batch processing
# top_queries_indices, top_documents_indices, top_queries_similarities, top_documents_similarities = perform_mips_batch(new_query_text, query_embeddings, document_embeddings, k=1000, lambda_param=0.5, query_batch_size=100, embedding_batch_size=100)


# print(len(top_queries_indices),len(top_documents_indices),len(top_queries_similarities),len(top_documents_similarities))
# print(len(top_queries_indices[0]),len(top_documents_indices[0]),len(top_queries_similarities[0]),len(top_documents_similarities[0]))


# results_df_dict = {}
# num_of_new_queries= len(new_query_ids)
# print("num_of_new_queries", num_of_new_queries)

# # Create DataFrames for Queries and Documents
# for new_qid, i_query, score_query, i_doc, score_doc in zip(new_query_ids, top_queries_indices, top_queries_similarities, top_documents_indices, top_documents_similarities):
#     # Create DataFrame for queries related to the current new_qid
#     queries_df = pd.DataFrame({
#         'new_qid': new_qid,
#         'sim_qid': train_queries_df.iloc[i_query]['qid'],
#         'sim_qtext': train_queries_df.iloc[i_query]['qtext'],
#         'q_sim_score': score_query
#     })

#     # Create DataFrame for documents related to the current new_qid
#     documents_df = pd.DataFrame({
#         'new_qid': new_qid,
#         'sim_docid': documentst_df.iloc[i_doc]['docid'],
#         'sim_doctext': documentst_df.iloc[i_doc]['doctext'],
#         'doc_sim_score': score_doc
#     })

#     results_df_dict[f'sim_queries_df_{new_qid}'] = queries_df
#     results_df_dict[f'sim_documents_df_{new_qid}'] = documents_df

    


def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores)[:k]
    # dcg = np.sum((2**relevance_scores - 1) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
    dcg = np.sum((relevance_scores) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
    return dcg


def ndcg_at_k(relevance_scores, k):
    dcg_max = dcg_at_k(sorted(relevance_scores, reverse=True), k)  # ideal DCG
    if not dcg_max:
        return 0
    return dcg_at_k(relevance_scores, k) / dcg_max

def dcg_at_k_v2(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores)[:k]
    dcg = np.sum((2**relevance_scores - 1) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
    # dcg = np.sum((relevance_scores) / np.log2(np.arange(1, len(relevance_scores) + 1) + 1))
    return dcg

def ndcg_at_k_v2(relevance_scores, k):
    dcg_max = dcg_at_k_v2(sorted(relevance_scores, reverse=True), k)  # ideal DCG
    if not dcg_max:
        return 0
    return dcg_at_k_v2(relevance_scores, k) / dcg_max


def mrr_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    for i, rel in enumerate(r):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(r, relevant_set, k):
    r = np.asarray(r, dtype=float)[:k]
    return np.sum(r > 0) / len(relevant_set) if len(relevant_set) > 0 else 0.0




# Final Score Dense Retriver (DR)

# for new_qid in new_query_ids:

#     sim_documents_df = results_df_dict[f'sim_documents_df_{new_qid}']

#     if sim_documents_df.shape[0]!=1000:
#         print(new_qid)

#     doc_scores = {}

#     # Step 1: Get dense retrieval score for each document in sim_documents_df
#     for index, row in sim_documents_df.iterrows():
#         doc_id = row['sim_docid']
#         dense_score = row['doc_sim_score']
        
#         # Add the dense retrieval score to the document's final score
#         if doc_id not in doc_scores:
#             doc_scores[doc_id] = 0
#         doc_scores[doc_id] += dense_score

#     # Store the final scores for this new_qid, sorted by 'final_score' in descending order
#     final_scores_df = pd.DataFrame.from_dict(doc_scores, orient='index', columns=['final_score']).reset_index().rename(columns={'index': 'docid'})

#     # Sort by 'final_score' in descending order
#     final_scores_df = final_scores_df.sort_values(by='final_score', ascending=False)
    
#     # Save the final score DataFrame for the current new_qid
#     results_df_dict[f'final_scores_df_{new_qid}'] = final_scores_df



# Save the dictionary to a file
# with open(r'resultsdf_DR_test_head', 'wb') as file:
#     pickle.dump(results_df_dict, file)
# print("Dictionary saved successfully.")


# Load the dictionary from the file
with open(r'resultsdf_DR_test_head', 'rb') as file:
    results_df_dict = pickle.load(file)
print("Dictionary loaded successfully.")



### DR
ndcg_scores_10 = []
ndcg_scores_1000 = []
ndcg_v2_scores_10 = []
ndcg_v2_scores_1000 = []
mrr_scores_10 = []
mrr_scores_1000 = []
recall_scores = []
recall_scores_10 =[]

# Top K documents to evaluate

for new_qid in new_query_ids:

    final_scores_df = results_df_dict[f'final_scores_df_{new_qid}']
    top_1000_docs = final_scores_df['docid'].values[:1000]

    # Get qrel scores for the top 10 documents
    top_1000_qrel = []
    for doc_id in top_1000_docs:
        if doc_id in raw_head_queries_test_df['docid'].values:
            qrel_value = raw_head_queries_test_df[(raw_head_queries_test_df['qid'] == new_qid) & 
                                                           (raw_head_queries_test_df['docid'] == doc_id)]['qrel'].values
            
            if len(qrel_value) > 0:
                top_1000_qrel.append(qrel_value[0])
            else:
                top_1000_qrel.append(0)
        else:
            top_1000_qrel.append(0)

    ### test queries
    relevant_docs = raw_head_queries_test_df[(raw_head_queries_test_df['qid'] == new_qid) & (raw_head_queries_test_df['qrel'] != 0)]

    ndcg_scores_10.append(ndcg_at_k(top_1000_qrel, 10))
    ndcg_scores_1000.append(ndcg_at_k(top_1000_qrel, 1000))
    ndcg_v2_scores_10.append(ndcg_at_k_v2(top_1000_qrel, 10))
    ndcg_v2_scores_1000.append(ndcg_at_k_v2(top_1000_qrel, 1000))
    mrr_scores_10.append(mrr_at_k(top_1000_qrel,10))
    mrr_scores_1000.append(mrr_at_k(top_1000_qrel,1000))
    recall_scores_10.append(recall_at_k(top_1000_qrel, relevant_docs['docid'].values, 10))
    recall_scores.append(recall_at_k(top_1000_qrel, relevant_docs['docid'].values, 1000))

# Calculate mean scores across all queries
mean_ndcg_10 = np.mean(ndcg_scores_10)
mean_ndcg_1000 = np.mean(ndcg_scores_1000)
mean_ndcg_10_v2 = np.mean(ndcg_v2_scores_10)
mean_ndcg_1000_v2 = np.mean(ndcg_v2_scores_1000)
mean_mrr_10 = np.mean(mrr_scores_10)
mean_mrr_1000 = np.mean(mrr_scores_1000)
mean_recall_10 = np.mean(recall_scores_10)
mean_recall = np.mean(recall_scores)

# Output the results

print("DR")
print(f'Mean NDCG@10: {mean_ndcg_10}')
print(f'Mean NDCG@1000: {mean_ndcg_1000}')
print(f'Mean NDCG_v2@10: {mean_ndcg_10_v2}')
print(f'Mean NDCG_v2@1000: {mean_ndcg_1000_v2}')
print(f'Mean MRR@10: {mean_mrr_10}')
print(f'Mean MRR@1000: {mean_mrr_1000}')
print(f'Mean Recall@10: {mean_recall_10}')
print(f'Mean Recall@1000: {mean_recall}')
print("\n")




### LA
# for new_qid in new_query_ids:

#     sim_queries_df = results_df_dict[f'sim_queries_df_{new_qid}']

#     doc_scores = {}

#     # Step 2: Compute log-augmentation score for documents in the relevant set (from qrel)
#     for index, row in sim_queries_df.iterrows():
#         sim_qid = row['sim_qid']
#         rel_docs = qrel_train_df[(qrel_train_df['qid'] == sim_qid) & (qrel_train_df['qrel'] != 0)]['docid'].dropna().values
#         q_sim_score = row['q_sim_score']
        
#         # Add the log-augmentation score to the document's final score if it is in Rel(q)
#         for doc_id in rel_docs:
#             if doc_id not in doc_scores:
#                 doc_scores[doc_id] = 0
#             # doc_scores[doc_id] += lambda_val * q_sim_score 
#             doc_scores[doc_id] += q_sim_score 

#     # Store the final scores for this new_qid, sorted by 'final_score' in descending order
#     final_scores_df = pd.DataFrame.from_dict(doc_scores, orient='index', columns=['final_score']).reset_index().rename(columns={'index': 'docid'})

#     # Sort by 'final_score' in descending order
#     final_scores_df = final_scores_df.sort_values(by='final_score', ascending=False)
    
#     # Save the final score DataFrame for the current new_qid
#     results_df_dict[f'final_scores_df_{new_qid}'] = final_scores_df

    

# Save the dictionary to a file
# with open(r'resultsdf_LA_test_head', 'wb') as file:
#     pickle.dump(results_df_dict, file)
# print("Dictionary saved successfully.")

# Load the dictionary from the file
with open(r'resultsdf_LA_test_head', 'rb') as file:
    results_df_dict = pickle.load(file)
print("Dictionary loaded successfully.")


### LA -->  choosing values[:1000]
ndcg_scores_10 = []
ndcg_scores_1000 = []
ndcg_v2_scores_10 = []
ndcg_v2_scores_1000 = []
mrr_scores_10 = []
mrr_scores_1000 = []
recall_scores = []
recall_scores_10 =[]

# Top K documents to evaluate

for new_qid in new_query_ids:

    final_scores_df = results_df_dict[f'final_scores_df_{new_qid}']
    top_1000_docs = final_scores_df['docid'].values[:1000]

    # Get qrel scores for the top 10 documents
    top_1000_qrel = []
    for doc_id in top_1000_docs:
        ### test queries 
        if doc_id in raw_head_queries_test_df['docid'].values:
            qrel_value = raw_head_queries_test_df[(raw_head_queries_test_df['qid'] == new_qid) & 
                                                           (raw_head_queries_test_df['docid'] == doc_id)]['qrel'].values
            
            if len(qrel_value) > 0:
                top_1000_qrel.append(qrel_value[0])
            else:
                top_1000_qrel.append(0)
        else:
            top_1000_qrel.append(0)


    relevant_docs = raw_head_queries_test_df[(raw_head_queries_test_df['qid'] == new_qid) & (raw_head_queries_test_df['qrel'] != 0)]


    ndcg_scores_10.append(ndcg_at_k(top_1000_qrel, 10))
    ndcg_scores_1000.append(ndcg_at_k(top_1000_qrel, 1000))
    ndcg_v2_scores_10.append(ndcg_at_k_v2(top_1000_qrel, 10))
    ndcg_v2_scores_1000.append(ndcg_at_k_v2(top_1000_qrel, 1000))
    mrr_scores_10.append(mrr_at_k(top_1000_qrel,10))
    mrr_scores_1000.append(mrr_at_k(top_1000_qrel,1000))
    recall_scores_10.append(recall_at_k(top_1000_qrel, relevant_docs['docid'].values, 10))
    recall_scores.append(recall_at_k(top_1000_qrel, relevant_docs['docid'].values, 1000))

# Calculate mean scores across all queries
mean_ndcg_10 = np.mean(ndcg_scores_10)
mean_ndcg_1000 = np.mean(ndcg_scores_1000)
mean_ndcg_10_v2 = np.mean(ndcg_v2_scores_10)
mean_ndcg_1000_v2 = np.mean(ndcg_v2_scores_1000)
mean_mrr_10 = np.mean(mrr_scores_10)
mean_mrr_1000 = np.mean(mrr_scores_1000)
mean_recall_10 = np.mean(recall_scores_10)
mean_recall = np.mean(recall_scores)

# Output the results

print("LA")
print(f'Mean NDCG@10: {mean_ndcg_10}')
print(f'Mean NDCG@1000: {mean_ndcg_1000}')
print(f'Mean NDCG_v2@10: {mean_ndcg_10_v2}')
print(f'Mean NDCG_v2@1000: {mean_ndcg_1000_v2}')
print(f'Mean MRR@10: {mean_mrr_10}')
print(f'Mean MRR@1000: {mean_mrr_1000}')
print(f'Mean Recall@10: {mean_recall_10}')
print(f'Mean Recall@1000: {mean_recall}')
print("\n")





### DR + LA
# lambda_val = 0.5 

# for new_qid in new_query_ids:

#     sim_queries_df = results_df_dict[f'sim_queries_df_{new_qid}']
#     sim_documents_df = results_df_dict[f'sim_documents_df_{new_qid}']

#     doc_scores = {}

#     # Step 1: Get dense retrieval score for each document in sim_documents_df
#     for index, row in sim_documents_df.iterrows():
#         doc_id = row['sim_docid']
#         dense_score = row['doc_sim_score']
        
#         # Add the dense retrieval score to the document's final score
#         if doc_id not in doc_scores:
#             doc_scores[doc_id] = 0
#         doc_scores[doc_id] += dense_score

#     # Step 2: Compute log-augmentation score for documents in the relevant set (from qrel)
#     for index, row in sim_queries_df.iterrows():
#         sim_qid = row['sim_qid']
#         rel_docs = qrel_train_df[(qrel_train_df['qid'] == sim_qid) & (qrel_train_df['qrel'] != 0)]['docid'].dropna().values
#         q_sim_score = row['q_sim_score']
        
#         # Add the log-augmentation score to the document's final score if it is in Rel(q)
#         for doc_id in rel_docs:
#             if doc_id not in doc_scores:
#                 doc_scores[doc_id] = 0
#             doc_scores[doc_id] += (lambda_val * q_sim_score )

#     # Store the final scores for this new_qid, sorted by 'final_score' in descending order
#     final_scores_df = pd.DataFrame.from_dict(doc_scores, orient='index', columns=['final_score']).reset_index().rename(columns={'index': 'docid'})

#     # Sort by 'final_score' in descending order
#     final_scores_df = final_scores_df.sort_values(by='final_score', ascending=False)
    
#     # Save the final score DataFrame for the current new_qid
#     results_df_dict[f'final_scores_df_{new_qid}'] = final_scores_df


# with open(r'resultsdf_DRplusLA_test_head', 'wb') as file:
#     pickle.dump(results_df_dict, file)
# print("Dictionary saved successfully.")

# Load the dictionary from the file
with open(r'resultsdf_DRplusLA_test_head', 'rb') as file:
    results_df_dict = pickle.load(file)
print("Dictionary loaded successfully.")


### DR+LA -->  choosing values[:1000]
ndcg_scores_10 = []
ndcg_scores_1000 = []
ndcg_v2_scores_10 = []
ndcg_v2_scores_1000 = []
mrr_scores_10 = []
mrr_scores_1000 = []
recall_scores = []
recall_scores_10 =[]

for new_qid in new_query_ids:

    final_scores_df = results_df_dict[f'final_scores_df_{new_qid}']
    top_1000_docs = final_scores_df['docid'].values[:1000]

    # Get qrel scores for the top 10 documents
    top_1000_qrel = []
    for doc_id in top_1000_docs:
        ### test queries 
        if doc_id in raw_head_queries_test_df['docid'].values:
            qrel_value = raw_head_queries_test_df[(raw_head_queries_test_df['qid'] == new_qid) & 
                                                           (raw_head_queries_test_df['docid'] == doc_id)]['qrel'].values
            
            if len(qrel_value) > 0:
                top_1000_qrel.append(qrel_value[0])
            else:
                top_1000_qrel.append(0)
        else:
            top_1000_qrel.append(0)


    relevant_docs = raw_head_queries_test_df[(raw_head_queries_test_df['qid'] == new_qid) & (raw_head_queries_test_df['qrel'] != 0)]


    ndcg_scores_10.append(ndcg_at_k(top_1000_qrel, 10))
    ndcg_scores_1000.append(ndcg_at_k(top_1000_qrel, 1000))
    ndcg_v2_scores_10.append(ndcg_at_k_v2(top_1000_qrel, 10))
    ndcg_v2_scores_1000.append(ndcg_at_k_v2(top_1000_qrel, 1000))
    mrr_scores_10.append(mrr_at_k(top_1000_qrel,10))
    mrr_scores_1000.append(mrr_at_k(top_1000_qrel,1000))
    recall_scores_10.append(recall_at_k(top_1000_qrel, relevant_docs['docid'].values, 10))
    recall_scores.append(recall_at_k(top_1000_qrel, relevant_docs['docid'].values, 1000))

# Calculate mean scores across all queries
mean_ndcg_10 = np.mean(ndcg_scores_10)
mean_ndcg_1000 = np.mean(ndcg_scores_1000)
mean_ndcg_10_v2 = np.mean(ndcg_v2_scores_10)
mean_ndcg_1000_v2 = np.mean(ndcg_v2_scores_1000)
mean_mrr_10 = np.mean(mrr_scores_10)
mean_mrr_1000 = np.mean(mrr_scores_1000)
mean_recall_10 = np.mean(recall_scores_10)
mean_recall = np.mean(recall_scores)

# Output the results

print("DR+LA")
print(f'Mean NDCG@10: {mean_ndcg_10}')
print(f'Mean NDCG@1000: {mean_ndcg_1000}')
print(f'Mean NDCG_v2@10: {mean_ndcg_10_v2}')
print(f'Mean NDCG_v2@1000: {mean_ndcg_1000_v2}')
print(f'Mean MRR@10: {mean_mrr_10}')
print(f'Mean MRR@1000: {mean_mrr_1000}')
print(f'Mean Recall@10: {mean_recall_10}')
print(f'Mean Recall@1000: {mean_recall}')
print("\n")


# Cheaking the statistically significant

# from scipy.stats import ttest_rel, wilcoxon


### DR

# ### NDCG@10
# # Perform paired t-test
# t_stat, p_value = ttest_rel(ModelA_DR_ndcg_scores_10, ModelB_DR_ndcg_scores_10)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")

# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_DR_ndcg_scores_10, ModelB_DR_ndcg_scores_10)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

# ### RR@10
# # Perform paired t-test
# t_stat, p_value = ttest_rel(ModelA_DR_mrr_scores_10, ModelB_DR_mrr_scores_10)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")

# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_DR_mrr_scores_10, ModelB_DR_mrr_scores_10)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

# ### Recall@1000
# # Perform paired t-test
# t_stat, p_value = ttest_rel(ModelA_DR_recall_scores, ModelB_DR_recall_scores)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")

# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_DR_recall_scores, ModelB_DR_recall_scores)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")


###LA

### NDCG@10

# # Perform paired t-test
# t_stat, p_value = ttest_rel(ModelA_LA_ndcg_scores_10, ModelB_LA_ndcg_scores_10)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")
# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_LA_ndcg_scores_10, ModelB_LA_ndcg_scores_10)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

# ### RR@10

# t_stat, p_value = ttest_rel(ModelA_LA_mrr_scores_10, ModelB_LA_mrr_scores_10)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")
# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_LA_mrr_scores_10, ModelB_LA_mrr_scores_10)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

# ### Recall@1000

# t_stat, p_value = ttest_rel(ModelA_LA_recall_scores, ModelB_LA_recall_scores)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")
# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_LA_recall_scores, ModelB_LA_recall_scores)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")


### DR+LA

### NDCG@10

# # Perform paired t-test
# t_stat, p_value = ttest_rel(ModelA_DR_and_LA_ndcg_scores_10, ModelB_DR_and_LA_ndcg_scores_10)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")
# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_DR_and_LA_ndcg_scores_10, ModelB_DR_and_LA_ndcg_scores_10)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

# ### RR@10

# t_stat, p_value = ttest_rel(ModelA_DR_and_LA_mrr_scores_10, ModelB_DR_and_LA_mrr_scores_10)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")
# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_DR_and_LA_mrr_scores_10, ModelB_DR_and_LA_mrr_scores_10)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

# ### Recall@1000

# t_stat, p_value = ttest_rel(ModelA_DR_and_LA_recall_scores, ModelB_DR_and_LA_recall_scores)
# print(f"Paired t-test: t_stat={t_stat:.4f}, p_value={p_value:.4f}")
# # Perform Wilcoxon signed-rank test
# w_stat, p_value_w = wilcoxon(ModelA_DR_and_LA_recall_scores, ModelB_DR_and_LA_recall_scores)
# print(f"Wilcoxon test: w_stat={w_stat:.4f}, p_value={p_value_w:.4f}")

