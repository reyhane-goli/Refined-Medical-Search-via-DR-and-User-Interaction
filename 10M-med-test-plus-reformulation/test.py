#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SBATCH directives (SLURM job configuration)
# These lines are comments and will be ignored by Python

#SBATCH --job-name=test   # Name of the job
#SBATCH --output=test.out    # Standard output log file
#SBATCH --error=test.err     # Standard error log file
#SBATCH --partition=bigmem                   # Request the bigmem partition (CPU)
#SBATCH --ntasks=1                           # Number of tasks (1 process)
#SBATCH --cpus-per-task=3                   # Number of CPU cores requested per task
#SBATCH --mem=200G                            # Amount of RAM requested
#SBATCH --time=00:45:00                      # Runtime (D-HH:MM:SS format)
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
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import psutil

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)


sys.stdout = open("test.txt", "w")
sys.stderr = sys.stdout  # Redirect stderr to the same file as stdout


def get_session_intersctions(sim_qid, sim_qtext):

    search_words = sim_qtext.split()

    folder_path = 'logs/'
    combined_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            year_str = filename.split('-')[0]
            if year_str in (['2018','2019','2020']):
                file_path = os.path.join(folder_path, filename)
                
                data = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
                
                df = pd.DataFrame(data)
                combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.drop(columns=['Url','DOI','ClinicalAreas','Title'],inplace=True)
    combined_df.rename(columns={'Keywords': 'QueryKeywords', 'DocumentId': 'ClickedDocumentIds'}, inplace=True)

    print("combined_df.shape[0]",combined_df.shape[0])

    # Filter rows where all words in search_words are in QueryKeywords
    filtered_df = combined_df[combined_df['QueryKeywords'].apply(
        lambda x: all(word in x for word in search_words)
    )]

    print("filtered_df.shape[0]",filtered_df.shape[0])

    # Find unique SessionIds from the filtered df
    unique_session_ids = filtered_df['SessionId'].unique()
    print("unique_session_ids",unique_session_ids)

    # Filter the main df to include all records with matching SessionIds
    session_filtered_df = combined_df[combined_df['SessionId'].isin(unique_session_ids)]
    clicked_document_ids = session_filtered_df['ClickedDocumentIds']
    clicked_document_ids_cleaned = (
    clicked_document_ids.dropna()  # Remove null values
        .apply(lambda x: int(x) if str(x).isdigit() else None)  # Ensure integer conversion
        .dropna()  # Remove any remaining invalid values
        .astype(int)  # Convert to integers
        .unique()  # Remove duplicates
    )

    session_docs = clicked_document_ids_cleaned.tolist()
    return session_docs

### Get Data - Documents
documentst_file = '../documents/docs.tsv'

# Read the TSV files
documentst_df = pd.read_csv(documentst_file, delimiter='\t', header=None, names=['docid', 'doctext'])


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


docid_to_title = dict(zip(documentst_df['docid'], documentst_df['title']))



sim_qid = 136
qtext="colon cancer screening"


########################### session - befor -after ##############################

with open('aggregated_before_after_clicks_2020.pkl', 'rb') as f:
    aggregate_data_2020 = pickle.load(f)

with open('aggregated_before_after_clicks_2019.pkl', 'rb') as f:
    aggregate_data_2019 = pickle.load(f)

with open('aggregated_before_after_clicks_2018.pkl', 'rb') as f:
    aggregate_data_2018 = pickle.load(f)

with open('aggregated_before_after_clicks_2017.pkl', 'rb') as f:
    aggregate_data_2017 = pickle.load(f)

aggregate_record_2020 = aggregate_data_2020[sim_qid]
aggregate_filtered_docs_2020 = [(docid, freq) for docid, freq in aggregate_record_2020.items() if docid in docid_to_title]
print(len(aggregate_filtered_docs_2020))
print(aggregate_filtered_docs_2020)
print("\n")
aggregate_record_2019 = aggregate_data_2019[sim_qid]
aggregate_filtered_docs_2019 = [(docid, freq) for docid, freq in aggregate_record_2019.items() if docid in docid_to_title]
print(len(aggregate_filtered_docs_2019))
print(aggregate_filtered_docs_2019)
print("\n")
aggregate_record_2018 = aggregate_data_2018[sim_qid]
aggregate_filtered_docs_2018 = [(docid, freq) for docid, freq in aggregate_record_2018.items() if docid in docid_to_title]
print(len(aggregate_filtered_docs_2018))
print(aggregate_filtered_docs_2018)
print("\n")
aggregate_record_2017 = aggregate_data_2017[sim_qid]
aggregate_filtered_docs_2017 = [(docid, freq) for docid, freq in aggregate_record_2017.items() if docid in docid_to_title]
print(len(aggregate_filtered_docs_2017))
print(aggregate_filtered_docs_2017)
print("\n")


aggregate_combined_records = {}

for docid, freq in aggregate_filtered_docs_2020:
    aggregate_combined_records[docid] = aggregate_combined_records.get(docid, 0) + freq

for docid, freq in aggregate_filtered_docs_2019:
    aggregate_combined_records[docid] = aggregate_combined_records.get(docid, 0) + freq

print(len(aggregate_combined_records))
print(aggregate_combined_records)

###################### qid_to_rel_docs.pkl #####################################

# with open('qid_to_rel_docs.pkl', 'rb') as f:
#     qid_to_rel_docs = pickle.load(f)


# print(len(qid_to_rel_docs))

# # Example: Retrieve rel_docs for a specific qid
# qid = sim_qid # Replace with the desired qid
# rel_docs = qid_to_rel_docs.get(qid, [])
# print(len(rel_docs))
# print(type(rel_docs))
# print(f"Relevant documents for qid {qid}: {rel_docs}")


###################### queries_with_click_counts.pkl ############################

# with open('queries_with_click_counts.pkl', 'rb') as f:
#     data = pickle.load(f)

# # Check the number of keys
# num_keys = len(data)
# print(f"The number of keys in the data is: {num_keys}")

# global_max_freq = max(
#     (freq for query_id in data if isinstance(data[query_id], dict) for freq in data[query_id].values()),
#     default=1  # Provide a default value if the sequence is empty
# )
# print(f"Global maximum frequency: {global_max_freq}")

# for query_id, value in data.items():
#     print(f"Query ID: {query_id}")
#     print(f"Type of value: {type(value)}")
    
#     if isinstance(value, Counter):
#         print("This is a Counter object.")
#     elif isinstance(value, dict):
#         print("This is a dictionary, but not a Counter.")
#     else:
#         print("This is NOT a dictionary.")



# dctr_head_queries_train_file = '../qrels/qrels.dctr.head.train.tsv'
# raw_head_queries_train_file = '../qrels/qrels.raw.head.train.tsv'
# raw_torso_queries_train_file = '../qrels/qrels.raw.torso.train.tsv'
# raw_tail_queries_train_file = '../qrels/qrels.raw.tail.train.tsv'


# # Read the TSV files
# dctr_head_queries_train_df = pd.read_csv(dctr_head_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# raw_head_queries_train_df = pd.read_csv(raw_head_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# raw_torso_queries_train_df = pd.read_csv(raw_torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# raw_tail_queries_train_df = pd.read_csv(raw_tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])



# raw_tail_queries_train_df = raw_tail_queries_train_df.drop(raw_tail_queries_train_df.columns[1], axis=1)
# raw_torso_queries_train_df = raw_torso_queries_train_df.drop(raw_torso_queries_train_df.columns[1], axis=1)
# raw_head_queries_train_df = raw_head_queries_train_df.drop(raw_head_queries_train_df.columns[1], axis=1)
# qrel_train_df = pd.concat([raw_head_queries_train_df, raw_torso_queries_train_df, raw_tail_queries_train_df], ignore_index=True)

# rel_docs = qrel_train_df[(qrel_train_df['qid'] == sim_qid) & (qrel_train_df['qrel'] != 0)]['docid'].dropna().values
# rel_docs = rel_docs.tolist()
# print(len(rel_docs))

# record = data[sim_qid]
# # filtered_docs = [(docid, freq) for docid, freq in record.items() if docid in docid_to_title]
# filtered_docs = [(docid, freq) for docid, freq in record.items()]
# docids = [docid for docid, _ in filtered_docs]
# print(len(docids))

# common_docids = set(docids).intersection(rel_docs)
# print(len(common_docids))

# max_freq = max_freq_dict.get(sim_qid, 1)
# doc_scores = {}
# for doc_id, freq in filtered_docs:
#     if doc_id not in doc_scores:
#         doc_scores[doc_id] = 0
#     # doc_scores[doc_id] += q_sim_score * (freq / max_freq)
#     doc_scores[doc_id] += freq

# print(doc_scores)

# Print the first 10 items with their types
# print("First 10 items in the data (with types):")
# for idx, (key, value) in enumerate(data.items()):
#     print(f"Key: {key} (Type: {type(key)})")
#     print(f"Value: {value} (Type: {type(value)})")
#     if idx == 9:  # Stop after printing the first 10 items
#         break

# print("\n")
# print("\n")

# from collections import Counter

# # Iterate over the dictionary or DataFrame to check the type of each value
# for key, value in data.items():
#     print(f"Key: {key}")
#     print(f"Value: {value} (Type: {type(value)})")
    
#     # Check if the value is a Counter
#     if isinstance(value, Counter):
#         print("The value is a Counter object.\n")
#     else:
#         print("The value is NOT a Counter object.\n")
    
#     # Stop after checking the first few entries
#     break

# print("\n")
# print("\n")





# Assuming `data` is already a DataFrame
# print(f"The number of rows in the DataFrame: {len(data)}")
# print(f"\nColumns in the DataFrame: {data.columns.tolist()}")

# for _, row in data.iterrows():
#     # Print qid
#     print(f"qid: {row['qid']}")
#     # Print all_clicks in the next line
#     print(f"all_clicks: {row['all_clicks']}\n")
    
#     # Stop after printing the first 10 rows
#     if row.name >= 9:
#         break


###################### work with aggregated_docid_counts.pkl ############################

with open('aggregated_docid_counts.pkl', 'rb') as f:
    data = pickle.load(f)

# num_keys = len(data)
# print("\n")
# print("\n")
# print(f"The number of keys in the data is: {num_keys}")

# Print the first 10 items with their types
print("First 10 items in the data (with types):")
for idx, (key, value) in enumerate(data.items()):
    print(f"Key: {key} (Type: {type(key)})")
    print(f"Value: {value} (Type: {type(value)})")
    if idx == 9:  # Stop after printing the first 10 items
        break




# Check the type of keys
# for key in data.keys():
#     print(f"Key: {key}, Type: {type(key)}")
#     break  # Check just the first key


# # Match session_docs with their titles and frequencies
# if sim_qid in data:
#     # Retrieve the record for this query ID
#     record = data[sim_qid]
#     print(f"Record for sim_qid {sim_qid} (qtext: '{qtext}'):")
#     print(record)
    
#     # Filter out documents without titles
#     filtered_docs = [(docid, freq) for docid, freq in record.items() if docid in docid_to_title]

#     # Sort by frequency (already provided by Counter, but this ensures consistency)
#     sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)

#     # Get the top 5 most common documents with titles
#     top_5_docs_with_titles = sorted_docs[:5]

#     # Display results with titles
#     print("\nTop 5 most common documents with titles (excluding 'Title not found'):")
#     for docid, freq in top_5_docs_with_titles:
#         title = docid_to_title[docid]
#         print(f"DocID: {docid}, Title: {title}, Frequency: {freq}")
# else:
#     print(f"sim_qid {sim_qid} not found in the data.")



######################## work with all clicks list ##############################

# train_queries_with_clicks = pd.read_pickle('train_queries_with_clicks.pkl')
# print("train_queries_with_clicks.shape[0]",train_queries_with_clicks.shape[0])

# session_docs = train_queries_with_clicks.loc[
#         train_queries_with_clicks['qid'] == sim_qid, 'all_clicks'].values

# if len(session_docs) > 0:
#     session_docs = session_docs[0]
# else:
#     session_docs = []

# print(len(session_docs))
# print(session_docs)



# # ### Get Data - Documents
# documentst_file = '../documents/docs.tsv'

# # Read the TSV files
# documentst_df = pd.read_csv(documentst_file, delimiter='\t', header=None, names=['docid', 'doctext'])


# def split_title_abstract(text):
#     # Check for the presence of markers
#     if '<eot>' in text:
#         title, abstract = text.split('<eot>', 1)
#     elif 'BACKGROUND :' in text:
#         title, abstract = text.split('BACKGROUND :', 1)
#     else:
#         # If neither marker is present, use the entire text as the title and leave abstract empty
#         return text, ''
    
#     # Clean up the title and abstract
#     title = title.strip()
#     abstract = abstract.strip()
    
#     return title, abstract

# # Apply the function to the `doctext` column
# documentst_df[['title', 'abstract']] = documentst_df['doctext'].apply(split_title_abstract).apply(pd.Series)


# docid_to_title = dict(zip(documentst_df['docid'], documentst_df['title']))

# # Match session_docs with their titles
# session_doc_titles = {docid: docid_to_title.get(docid, "Title not found") for docid in session_docs}

# # Print or process the results
# # print(session_doc_titles)

# filtered_session_doc_titles = {docid: title for docid, title in session_doc_titles.items() if title != "Title not found"}

# # Count how many docids remain
# remaining_count = len(filtered_session_doc_titles)

# # Print or process the results
# print(f"Number of docids with valid titles: {remaining_count}")


# # docid_set = set(documentst_df['docid'])
# # filtered_session_docs = [doc for doc in session_docs if doc in docid_set]
# # print(f"Number of filtered session docs: {len(filtered_session_docs)}")



# # Print each docid and its title on a new line
# # for docid, title in filtered_session_doc_titles.items():
# #     print(f"DocID: {docid}, Title: {title}")



# # # Get Data Query Relevant Score
# # # queries File paths
# # dctr_head_queries_train_file = '../qrels/qrels.dctr.head.train.tsv'
# # dctr_head_queries_val_file = '../qrels/qrels.dctr.head.val.tsv'
# # dctr_head_queries_test_file = '../qrels/qrels.dctr.head.test.tsv'

# raw_head_queries_train_file = '../qrels/qrels.raw.head.train.tsv'
# # raw_head_queries_val_file = '../qrels/qrels.raw.head.val.tsv'
# # raw_head_queries_test_file = '../qrels/qrels.raw.head.test.tsv'

# # raw_torso_queries_train_file = '../qrels/qrels.raw.torso.train.tsv'
# # raw_torso_queries_val_file = '../qrels/qrels.raw.torso.val.tsv'
# # raw_torso_queries_test_file = '../qrels/qrels.raw.torso.test.tsv'

# # raw_tail_queries_train_file = '../qrels/qrels.raw.tail.train.tsv'
# # raw_tail_queries_val_file = '../qrels/qrels.raw.tail.val.tsv'
# # raw_tail_queries_test_file = '../qrels/qrels.raw.tail.test.tsv'

# # # Read the TSV files
# # dctr_head_queries_train_df = pd.read_csv(dctr_head_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # dctr_head_queries_val_df = pd.read_csv(dctr_head_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # dctr_head_queries_test_df = pd.read_csv(dctr_head_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])

# raw_head_queries_train_df = pd.read_csv(raw_head_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # raw_head_queries_val_df = pd.read_csv(raw_head_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # raw_head_queries_test_df = pd.read_csv(raw_head_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])

# # raw_torso_queries_train_df = pd.read_csv(raw_torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # raw_torso_queries_val_df = pd.read_csv(raw_torso_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # raw_torso_queries_test_df = pd.read_csv(raw_torso_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])

# # raw_tail_queries_train_df = pd.read_csv(raw_tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # raw_tail_queries_val_df = pd.read_csv(raw_tail_queries_val_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])
# # raw_tail_queries_test_df = pd.read_csv(raw_tail_queries_test_file, delimiter='\t', header=None, names=['qid', 'temp', 'docid', 'qrel'])


# # dctr_head_queries_train_df = dctr_head_queries_train_df.drop(dctr_head_queries_train_df.columns[1], axis=1)
# # dctr_head_queries_val_df = dctr_head_queries_val_df.drop(dctr_head_queries_val_df.columns[1], axis=1)
# # dctr_head_queries_test_df = dctr_head_queries_test_df.drop(dctr_head_queries_test_df.columns[1], axis=1)

# raw_head_queries_train_df = raw_head_queries_train_df.drop(raw_head_queries_train_df.columns[1], axis=1)
# # raw_head_queries_val_df = raw_head_queries_val_df.drop(raw_head_queries_val_df.columns[1], axis=1)
# # raw_head_queries_test_df = raw_head_queries_test_df.drop(raw_head_queries_test_df.columns[1], axis=1)

# # raw_torso_queries_train_df = raw_torso_queries_train_df.drop(raw_torso_queries_train_df.columns[1], axis=1)
# # raw_torso_queries_val_df = raw_torso_queries_val_df.drop(raw_torso_queries_val_df.columns[1], axis=1)
# # raw_torso_queries_test_df = raw_torso_queries_test_df.drop(raw_torso_queries_test_df.columns[1], axis=1)

# # raw_tail_queries_train_df = raw_tail_queries_train_df.drop(raw_tail_queries_train_df.columns[1], axis=1)
# # raw_tail_queries_val_df = raw_tail_queries_val_df.drop(raw_tail_queries_val_df.columns[1], axis=1)
# # raw_tail_queries_test_df =raw_tail_queries_test_df.drop(raw_tail_queries_test_df.columns[1], axis=1)

# # qrel_train_df = pd.concat([raw_head_queries_train_df, raw_torso_queries_train_df, raw_tail_queries_train_df], ignore_index=True)
# # rel_docs = qrel_train_df[(qrel_train_df['qid'] == sim_qid) & (qrel_train_df['qrel'] != 0)]['docid'].dropna().values
# # print(rel_docs)


# rel_docs = raw_head_queries_train_df[(raw_head_queries_train_df['qid'] == sim_qid) & (raw_head_queries_train_df['qrel'] != 0)]['docid'].dropna().values

# # Find the common docids
# common_docids = set(filtered_session_doc_titles.keys()).intersection(rel_docs)

# # Print the number of common docids
# print(f"Number of common docids: {len(common_docids)}")

# # Print each common docid, its title, and total count
# for docid in common_docids:
#     print(f"DocID: {docid}, Title: {filtered_session_doc_titles[docid]}")


# # Combine query and document titles for TF-IDF
# all_texts = [qtext] + list(filtered_session_doc_titles.values())

# # Compute TF-IDF features
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(all_texts)

# # Calculate cosine similarities between query and document titles
# similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# # Pair similarities with document IDs
# doc_similarities = {docid: sim for docid, sim in zip(filtered_session_doc_titles.keys(), similarities)}

# # Sort documents by similarity score
# sorted_docs = sorted(doc_similarities.items(), key=lambda x: x[1], reverse=True)

# # Display sorted results
# print("Document similarities to query:")
# for docid, sim in sorted_docs:
#     print(f"DocID: {docid}, Similarity: {sim:.4f}")




