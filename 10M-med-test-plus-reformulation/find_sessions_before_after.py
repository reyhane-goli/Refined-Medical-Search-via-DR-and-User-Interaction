#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SBATCH directives (SLURM job configuration)
# These lines are comments and will be ignored by Python

#SBATCH --job-name=find_session_befor_after
#SBATCH --output=find_session_befor_after.out
#SBATCH --error=find_session_befor_after.err
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=700G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rgoli@student.unimelb.edu.au


import os
import json
import dask.dataframe as dd  # Faster than pandas for large data
import re
import multiprocessing as mp  # Replacing Ray
from collections import Counter
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import pickle
import sys
import logging
import numpy as np

# --cpus-per-task=100
# --mem=600G
# --time=3-00:00:00

sys.stdout = open("find_session_befor_after_multi_cpu-2013.txt", "w")
sys.stderr = sys.stdout  # Redirect stderr to the same file as stdout

# Convert DateCreated to datetime
def convert_to_datetime(date_str):
    timestamp = int(date_str[6:-2]) / 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

# Process logs in parallel using Dask
# def process_log_file(file_path):
#     df = dd.read_json(file_path, lines=True)
#     df["DateCreated"] = df["DateCreated"].apply(convert_to_datetime, meta=("DateCreated", "datetime64[ns]"))
#     df = df[["SessionId", "QueryKeywords", "ClickedDocumentIds", "DateCreated"]].compute()
#     return df

def process_log_file(file_path):
    df = dd.read_json(file_path, lines=True)

    required_columns = ["SessionId", "Keywords", "DocumentId", "DateCreated"]
    existing_columns = [col for col in required_columns if col in df.columns]
    
    if not existing_columns:
        raise ValueError(f"None of the required columns found in {file_path}")

    df = df[existing_columns] 

    rename_map = {
        "Keywords": "QueryKeywords",
        "DocumentId": "ClickedDocumentIds"
    }
    rename_map = {orig: new for orig, new in rename_map.items() if orig in df.columns}
    df = df.rename(columns=rename_map)

    if "DateCreated" in df.columns:
        df["DateCreated"] = df["DateCreated"].apply(convert_to_datetime, meta=("DateCreated", "datetime64[ns]"))

    df = df.compute()

    final_required = ["SessionId", "QueryKeywords", "ClickedDocumentIds", "DateCreated"]
    missing_columns = [col for col in final_required if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing expected columns after processing: {missing_columns}")

    return df


# Load logs efficiently in parallel
def load_logs_parallel(folder_path="logs/"):

    log_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.endswith(".json") and f.startswith("2013-")]

    if not log_files:
        raise ValueError("No log files found for the year 2019 in the directory.")

    # Use Dask for faster loading
    ddf = dd.concat([process_log_file(f) for f in log_files])
    combined_df = ddf.compute()  # Convert to pandas for efficient lookups

    # Sort sessions by DateCreated to ensure correct before/after assignment
    session_dict = {
        session_id: session_df.sort_values(by="DateCreated").reset_index(drop=True)
        for session_id, session_df in combined_df.groupby("SessionId")
    }
    
    return session_dict

# Function to find before and after clicks
# def find_before_after_clicks(qtext, session_df):
#     query_indices = session_df[session_df["QueryKeywords"].str.contains(re.escape(qtext), na=False)].index
#     aggregated_clicks = Counter()

#     for idx in query_indices:
#         current_time = session_df.iloc[idx]["DateCreated"]
#         session_length = len(session_df)

#         # Case 1: First query in session (No before-query, check after-query only)
#         if idx == 0:
#             if session_length > 1:  # Ensure there is a next query
#                 next_time = session_df.iloc[idx + 1]["DateCreated"]
#                 if next_time - current_time <= timedelta(days=1):
#                     next_clicks = session_df.iloc[idx + 1]["ClickedDocumentIds"]
#                     if isinstance(next_clicks, str):
#                         next_clicks = json.loads(next_clicks)
#                     elif next_clicks is None:
#                         next_clicks = []
#                     aggregated_clicks.update(next_clicks)

#         # Case 2: Last query in session (No after-query, check before-query only)
#         elif idx == session_length - 1:
#             prev_time = session_df.iloc[idx - 1]["DateCreated"]
#             if current_time - prev_time <= timedelta(days=1):
#                 prev_clicks = session_df.iloc[idx - 1]["ClickedDocumentIds"]
#                 if isinstance(prev_clicks, str):
#                     prev_clicks = json.loads(prev_clicks)
#                 elif prev_clicks is None:
#                     prev_clicks = []
#                 aggregated_clicks.update(prev_clicks)

#         # Case 3: Middle queries (Check both before and after)
#         else:
#             prev_time = session_df.iloc[idx - 1]["DateCreated"]
#             if current_time - prev_time <= timedelta(days=1):
#                 prev_clicks = session_df.iloc[idx - 1]["ClickedDocumentIds"]
#                 if isinstance(prev_clicks, str):
#                     prev_clicks = json.loads(prev_clicks)
#                 elif prev_clicks is None:
#                     prev_clicks = []
#                 aggregated_clicks.update(prev_clicks)

#             next_time = session_df.iloc[idx + 1]["DateCreated"]
#             if next_time - current_time <= timedelta(days=1):
#                 next_clicks = session_df.iloc[idx + 1]["ClickedDocumentIds"]
#                 if isinstance(next_clicks, str):
#                     next_clicks = json.loads(next_clicks)
#                 elif next_clicks is None:
#                     next_clicks = []
#                 aggregated_clicks.update(next_clicks)

#     return aggregated_clicks




def find_before_after_clicks(qtext, session_df):
    query_indices = session_df[session_df["QueryKeywords"].str.contains(re.escape(qtext), na=False)].index
    aggregated_clicks = Counter()

    for idx in query_indices:
        current_time = session_df.iloc[idx]["DateCreated"]
        session_length = len(session_df)

        # Function to safely extract clicks as a list
        def extract_clicks(clicks):
            if isinstance(clicks, str):  # JSON format
                return json.loads(clicks)
            elif isinstance(clicks, (int, float, np.int64, np.float64)):  # Single number
                return [clicks]
            elif isinstance(clicks, list):  # Already a list
                return clicks
            return []  # Default empty list

        # Case 1: First query in session (Only check after-query)
        if idx == 0:
            if session_length > 1:  # Ensure there's a next query
                next_clicks = extract_clicks(session_df.iloc[idx + 1]["ClickedDocumentIds"])
                aggregated_clicks.update(next_clicks)

        # Case 2: Last query in session (Only check before-query)
        elif idx == session_length - 1:
            prev_clicks = extract_clicks(session_df.iloc[idx - 1]["ClickedDocumentIds"])
            aggregated_clicks.update(prev_clicks)

        # Case 3: Middle queries (Check both before and after)
        else:
            prev_clicks = extract_clicks(session_df.iloc[idx - 1]["ClickedDocumentIds"])
            next_clicks = extract_clicks(session_df.iloc[idx + 1]["ClickedDocumentIds"])

            aggregated_clicks.update(prev_clicks)
            aggregated_clicks.update(next_clicks)

    return aggregated_clicks

# Function to process a single query
def process_query(row, session_dict):
    qid = row["qid"]
    qtext = row["qtext"]
    aggregated_clicks = Counter()

    # Filter only relevant sessions containing the query
    relevant_sessions = {sid: s_df for sid, s_df in session_dict.items() if qtext in s_df["QueryKeywords"].values}

    for session_id, session_df in relevant_sessions.items():
        clicks = find_before_after_clicks(qtext, session_df)
        aggregated_clicks.update(clicks)

    return qid, aggregated_clicks

# Multiprocessing wrapper function
def parallel_process_queries(queries, session_dict, num_workers=60):
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.starmap(process_query, [(row, session_dict) for row in queries]), total=len(queries)))
    return results

if __name__ == "__main__":
    # Load logs using parallel processing
    print("Loading log data...")
    session_dict = load_logs_parallel()  # 

    print("Session dictionary created (sorted by DateCreated).")

    # Load queries
    print("Loading queries...")
    train_queries_df = dd.concat([
        dd.read_csv("../queries/topics.head.train.tsv", delimiter="\t", header=None, names=["qid", "qtext"]),
        dd.read_csv("../queries/topics.torso.train.tsv", delimiter="\t", header=None, names=["qid", "qtext"]),
        dd.read_csv("../queries/topics.tail.train.tsv", delimiter="\t", header=None, names=["qid", "qtext"])
    ]).compute()

    data_for_processing = train_queries_df.to_dict("records")

    # Parallel query processing using multiprocessing
    print("Processing queries in parallel...")
    results = parallel_process_queries(data_for_processing, session_dict, num_workers=60)

    # Save results
    aggregated_results = {qid: clicks for qid, clicks in results}
    with open("aggregated_before_after_clicks_2013.pkl", "wb") as f:
        pickle.dump(aggregated_results, f)

    print("Results saved as 'aggregated_before_after_clicks.pkl'")

# # Convert DateCreated to datetime
# def convert_to_datetime(date_str):
#     timestamp = int(date_str[6:-2]) / 1000
#     # return datetime.utcfromtimestamp(timestamp)
#     return datetime.fromtimestamp(timestamp, tz=timezone.utc)


# # Load and preprocess the combined DataFrame
# def load_logs(folder_path='logs/'):
#     combined_df = pd.DataFrame()
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.json'):
#             year_str = filename.split('-')[0]
#             # if year_str in ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']:
#             if year_str in ['2016', '2017', '2018', '2019', '2020']:
#                 file_path = os.path.join(folder_path, filename)
#                 data = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
#                 df = pd.DataFrame(data)
#                 combined_df = pd.concat([combined_df, df], ignore_index=True)

#     combined_df.drop(columns=['Url', 'DOI', 'ClinicalAreas', 'Title'], inplace=True)
#     combined_df.rename(columns={'Keywords': 'QueryKeywords', 'DocumentId': 'ClickedDocumentIds'}, inplace=True)
#     combined_df['DateCreated'] = combined_df['DateCreated'].apply(convert_to_datetime)
#     combined_df.sort_values(by=['SessionId', 'DateCreated'], inplace=True)
    
#     return combined_df

# # Function to find before and after clicks within a 1-day time difference
# def find_before_after_clicks(qtext, session_df):
#     query_indices = session_df[session_df['QueryKeywords'].str.contains(re.escape(qtext), na=False)].index
#     aggregated_clicks = Counter()

#     for idx in query_indices:
#         current_time = session_df.iloc[idx]['DateCreated']
#         session_length = len(session_df)

#         if idx > 0:  # Check the query before
#             prev_time = session_df.iloc[idx - 1]['DateCreated']
#             if current_time - prev_time <= timedelta(days=1):
#                 prev_clicks = session_df.iloc[idx - 1]['ClickedDocumentIds']
#                 aggregated_clicks.update(prev_clicks if isinstance(prev_clicks, list) else [])

#         if idx < session_length - 1:  # Check the query after
#             next_time = session_df.iloc[idx + 1]['DateCreated']
#             if next_time - current_time <= timedelta(days=1):
#                 next_clicks = session_df.iloc[idx + 1]['ClickedDocumentIds']
#                 aggregated_clicks.update(next_clicks if isinstance(next_clicks, list) else [])

#     return aggregated_clicks

# # Process each query in parallel
# def process_query(row, session_dict):
#     qid = row['qid']
#     qtext = row['qtext']
#     aggregated_clicks = Counter()

#     for session_id, session_df in session_dict.items():
#         clicks = find_before_after_clicks(qtext, session_df)
#         aggregated_clicks.update(clicks)

#     return qid, aggregated_clicks

# # Wrapper function for multiprocessing
# def parallel_process_queries(queries, session_dict, num_workers):
#     with mp.Pool(num_workers) as pool:
#         results = list(tqdm(pool.starmap(process_query, [(row, session_dict) for row in queries]), total=len(queries)))
#     return results

# if __name__ == "__main__":
#     # Load session data
#     print("Loading log data...")
#     combined_df = load_logs()

#     # Convert session data into a dictionary for efficient lookup
#     session_dict = {session_id: session_df.sort_values(by='DateCreated').reset_index(drop=True)
#                     for session_id, session_df in combined_df.groupby('SessionId')}
    
#     print("Session dictionary created.")

#     # Load queries
#     print("Loading queries...")
#     head_queries_train_file = '../queries/topics.head.train.tsv'
#     torso_queries_train_file = '../queries/topics.torso.train.tsv'
#     tail_queries_train_file = '../queries/topics.tail.train.tsv'

#     head_queries_train_df = pd.read_csv(head_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
#     torso_queries_train_df = pd.read_csv(torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
#     tail_queries_train_df = pd.read_csv(tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
    
#     train_queries_df = pd.concat([head_queries_train_df, torso_queries_train_df, tail_queries_train_df])
#     data_for_processing = train_queries_df.to_dict('records')

#     # Get number of available CPUs
#     num_workers = mp.cpu_count()
#     print(f"Using {num_workers} CPUs for parallel processing.")

#     # Parallel processing
#     print("Processing queries in parallel...")
#     results = parallel_process_queries(data_for_processing, session_dict, num_workers)

#     # Save results
#     aggregated_results = {qid: clicks for qid, clicks in results}

#     with open('aggregated_before_after_clicks.pkl', 'wb') as f:
#         pickle.dump(aggregated_results, f)

#     print("Results saved as 'aggregated_before_after_clicks.pkl'")




# # Load the combined DataFrame
# folder_path = 'logs/'
# combined_df = pd.DataFrame()
# for filename in os.listdir(folder_path):
#     if filename.endswith('.json'):
#         year_str = filename.split('-')[0]
#         if year_str in (['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']):
#             file_path = os.path.join(folder_path, filename)
#             data = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
#             df = pd.DataFrame(data)
#             combined_df = pd.concat([combined_df, df], ignore_index=True)

# combined_df.drop(columns=['Url', 'DOI', 'ClinicalAreas', 'Title'], inplace=True)
# combined_df.rename(columns={'Keywords': 'QueryKeywords', 'DocumentId': 'ClickedDocumentIds'}, inplace=True)
# combined_df['DateCreated'] = combined_df['DateCreated'].apply(convert_to_datetime)
# combined_df.sort_values(by=['SessionId', 'DateCreated'], inplace=True)

# # Function to find before and after clicks within a 1-day time difference
# def find_before_after_clicks(qtext, session_df):

#     query_indices = session_df[session_df['QueryKeywords'].str.contains(re.escape(qtext), na=False)].index
#     aggregated_clicks = Counter()

#     for idx in query_indices:
#         current_time = session_df.iloc[idx]['DateCreated']

#         session_length = len(session_df)

#         # If this is the first query in the session, only consider the next query
#         if idx == 0 and session_length > 1:
#             next_time = session_df.iloc[idx + 1]['DateCreated']
#             if next_time - current_time <= timedelta(days=1):
#                 next_clicks = session_df.iloc[idx + 1]['ClickedDocumentIds']
#                 aggregated_clicks.update(next_clicks if isinstance(next_clicks, list) else [])

#         # If this is the last query in the session, only consider the previous query
#         elif idx == session_length - 1 and session_length > 1:
#             prev_time = session_df.iloc[idx - 1]['DateCreated']
#             if current_time - prev_time <= timedelta(days=1):
#                 prev_clicks = session_df.iloc[idx - 1]['ClickedDocumentIds']
#                 aggregated_clicks.update(prev_clicks if isinstance(prev_clicks, list) else [])

#         # Otherwise, check both previous and next queries
#         else:
#             if idx > 0:  # Check the query before
#                 prev_time = session_df.iloc[idx - 1]['DateCreated']
#                 if current_time - prev_time <= timedelta(days=1):
#                     prev_clicks = session_df.iloc[idx - 1]['ClickedDocumentIds']
#                     aggregated_clicks.update(prev_clicks if isinstance(prev_clicks, list) else [])

#             if idx < session_length - 1:  # Check the query after
#                 next_time = session_df.iloc[idx + 1]['DateCreated']
#                 if next_time - current_time <= timedelta(days=1):
#                     next_clicks = session_df.iloc[idx + 1]['ClickedDocumentIds']
#                     aggregated_clicks.update(next_clicks if isinstance(next_clicks, list) else [])

#     return aggregated_clicks

# # Process each query
# def process_query(row):
#     qid = row['qid']
#     qtext = row['qtext']
#     aggregated_clicks = Counter()

#     # Process each session
#     for session_id, session_df in combined_df.groupby('SessionId'):
#         session_df = session_df.sort_values(by='DateCreated').reset_index(drop=True)
#         clicks = find_before_after_clicks(qtext, session_df)
#         aggregated_clicks.update(clicks)

#     print(f"Processed qid: {qid}")
#     return aggregated_clicks

# if __name__ == "__main__":
#     # Load queries
#     head_queries_train_file = '../queries/topics.head.train.tsv'
#     head_queries_train_df = pd.read_csv(head_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
#     torso_queries_train_file = '../queries/topics.torso.train.tsv'
#     torso_queries_train_df = pd.read_csv(torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
#     tail_queries_train_file = '../queries/topics.tail.train.tsv'
#     tail_queries_train_df = pd.read_csv(tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
#     train_queries_df = pd.concat([head_queries_train_df, torso_queries_train_df, tail_queries_train_df])

#     # Prepare for processing
#     data_for_processing = train_queries_df.to_dict('records')

#     # Process queries
#     results = []
#     for row in tqdm(data_for_processing):
#         results.append(process_query(row))

#     train_queries_df['aggregated_before_after_clicks'] = results

#     # Save results to a pkl file
#     aggregated_results = {}
#     for qid, clicks in zip(train_queries_df['qid'], train_queries_df['aggregated_before_after_clicks']):
#         aggregated_results[qid] = clicks

#     with open('aggregated_before_after_clicks.pkl', 'wb') as f:
#         pickle.dump(aggregated_results, f)

#     print("Results saved as 'aggregated_before_after_clicks.pkl'")
