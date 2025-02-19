#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SBATCH directives (SLURM job configuration)
# These lines are comments and will be ignored by Python

#SBATCH --job-name=find_session_clickweight
#SBATCH --output=find_session_clickweight.out
#SBATCH --error=find_session_clickweight.err
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --time=00:05:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rgoli@student.unimelb.edu.au

import pickle


with open('train_queries_with_click_counts.pkl', 'rb') as f:
    data = pickle.load(f)

# Filter the DataFrame to keep only 'qid' and 'all_clicks'
filtered_data = data[['qid', 'all_clicks']]

# Convert the filtered DataFrame into a dictionary for compatibility with your script
filtered_dict = {
    row['qid']: row['all_clicks']
    for _, row in filtered_data.iterrows()
}

# Save the dictionary to a pickle file
output_file = 'queries_with_click_counts.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(filtered_dict, f)

print(f"Filtered data saved to {output_file}.")
