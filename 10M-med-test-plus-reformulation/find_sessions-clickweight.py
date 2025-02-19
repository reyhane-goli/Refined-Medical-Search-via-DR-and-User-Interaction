#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SBATCH directives (SLURM job configuration)
# These lines are comments and will be ignored by Python

#SBATCH --job-name=find_session_clickweight
#SBATCH --output=find_session_clickweight.out
#SBATCH --error=find_session_clickweight.err
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rgoli@student.unimelb.edu.au

import pickle
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager , Lock
import logging
from tqdm import tqdm
import sys
import re
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("find_session_clickweight.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger()

# Load the combined DataFrame
logger.info("Loading combined DataFrame...")
folder_path = 'logs/'
combined_df = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        year_str = filename.split('-')[0]
        if year_str in (['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']):
            file_path = os.path.join(folder_path, filename)
            data = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
            df = pd.DataFrame(data)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.drop(columns=['Url', 'DOI', 'ClinicalAreas', 'Title'], inplace=True)
combined_df.rename(columns={'Keywords': 'QueryKeywords', 'DocumentId': 'ClickedDocumentIds'}, inplace=True)
logger.info(f"Loaded combined DataFrame with {combined_df.shape[0]} rows.")

head_queries_train_file = '../queries/topics.head.train.tsv'
head_queries_train_df = pd.read_csv(head_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
torso_queries_train_file = '../queries/topics.torso.train.tsv'
torso_queries_train_df = pd.read_csv(torso_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
tail_queries_train_file = '../queries/topics.tail.train.tsv'
tail_queries_train_df = pd.read_csv(tail_queries_train_file, delimiter='\t', header=None, names=['qid', 'qtext'])
train_queries_df = pd.concat([head_queries_train_df, torso_queries_train_df, tail_queries_train_df])
logger.info(f"Loaded train queries DataFrame with {train_queries_df.shape[0]} rows.")

# Ensure ClickedDocumentIds is an integer type
combined_df['ClickedDocumentIds'] = combined_df['ClickedDocumentIds'].astype(int)

# Group combined_df by SessionId for quick access
# session_clicks_map = combined_df.groupby('SessionId')['ClickedDocumentIds'].apply(list).to_dict()
# logger.info("Grouped combined DataFrame by SessionId for quick access.")

# Function to find all clicks for a given query
def get_query_specific_click_counts(query):
    """
    Find all clicked documents and their click counts for the given query.
    Returns a dictionary where keys are document IDs and values are their click counts.
    """
    # Escape special characters in the query text
    escaped_query = re.escape(query)

    # Find rows where QueryKeywords exactly match the query
    matching_rows = combined_df[combined_df['QueryKeywords'].str.match(f"^{escaped_query}$", na=False)]
    
    # Aggregate clicks for these rows
    all_clicks = matching_rows['ClickedDocumentIds'].tolist()
    click_counts = Counter(all_clicks)  # Count occurrences of each document ID
    
    return dict(click_counts)  # Convert Counter object to dictionary


# Parallel processing function with logging
def process_query(row):
    qid = row['qid']
    qtext = row['qtext']
    result = get_query_specific_click_counts(qtext)
    
    # Log progress by printing qid
    print(f"Processed qid: {qid}")
    
    return result

if __name__ == "__main__":
    num_cpus = cpu_count()
    print(f"Using {num_cpus} CPUs for parallel processing...")

    # Prepare data for multiprocessing
    data_for_processing = train_queries_df.to_dict('records')  # Each row as a dictionary

    # Use multiprocessing Pool
    with Pool(processes=num_cpus) as pool:
        # Use tqdm to display a progress bar
        results = list(tqdm(pool.imap(process_query, data_for_processing), total=len(data_for_processing)))

    # Add results to the DataFrame
    train_queries_df['all_clicks'] = results


    # Save the resulting DataFrame
    train_queries_df.to_pickle('train_queries_with_click_counts.pkl')
    print("Processed DataFrame saved as 'queries_with_click_counts.pkl'")

    # V1
    # # Save the resulting DataFrame
    # train_queries_df.to_pickle('train_queries_with_clicks.pkl')
    # print("Processed DataFrame saved as 'train_queries_with_clicks.pkl'")

    # V2
    # aggregated_results = {}
    # for qid, docid_count in zip(train_queries_df['qid'], train_queries_df['all_clicks']):
    #     if qid not in aggregated_results:
    #         aggregated_results[qid] = Counter()
    #     aggregated_results[qid].update(docid_count)

    # # Save the aggregated results
    # with open('aggregated_docid_counts.pkl', 'wb') as f:
    #     pickle.dump(aggregated_results, f)
    
    # print("Aggregated results saved as 'aggregated_docid_counts.pkl'")



# import pickle

# # Filter the DataFrame to keep only 'qid' and 'all_clicks'
# filtered_data = data[['qid', 'all_clicks']]

# # Convert the filtered DataFrame into a dictionary for compatibility with your script
# filtered_dict = {
#     row['qid']: row['all_clicks']
#     for _, row in filtered_data.iterrows()
# }

# # Save the dictionary to a pickle file
# output_file = 'qid_all_clicks.pkl'
# with open(output_file, 'wb') as f:
#     pickle.dump(filtered_dict, f)

# print(f"Filtered data saved to {output_file}.")
