import json
import logging
import math
import torch
logger = logging.getLogger(__name__)

class TripletDataset(torch.utils.data.Dataset):
	def __init__(self, dataset_path):
		'''
		dataset_path: path to a training jsonline file
		'''
		self.examples = []

		with open(dataset_path, "r", encoding="utf-8") as f:
			lines = f.readlines()

		for line in lines:
			entry = json.loads(line)
			entry['qid'] = str(entry['qid'])
			entry['pos_pmid'] = str(entry['pos_pmid'])
			entry['neg_pmid'] = str(entry['neg_pmid'])
			self.examples.append(entry)
	
	def __getitem__(self, index):
		entry = self.examples[index]
		return entry['qid'], entry['pos_pmid'], entry['neg_pmid']
	
	def __len__(self):
		return len(self.examples)
