import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class Biencoder(nn.Module): 
    def __init__(self, args):
        super(Biencoder, self).__init__()
        self.args = args

        q_path = args.bert_q_path
        d_path = args.bert_d_path

        self.config_q = AutoConfig.from_pretrained(q_path)
        self.bert_q = AutoModel.from_pretrained(q_path)

        self.config_d = AutoConfig.from_pretrained(d_path)
        self.bert_d = AutoModel.from_pretrained(d_path)


    def save_pretrained(self, path):
        self.config_q.save_pretrained(os.path.join(path, 'query_encoder'))
        self.bert_q.save_pretrained(os.path.join(path, 'query_encoder'))

        self.config_d.save_pretrained(os.path.join(path, 'doc_encoder'))
        self.bert_d.save_pretrained(os.path.join(path, 'doc_encoder'))


    def forward(self, q_input_ids, q_token_type_ids, q_attention_mask,pos_d_input_ids, pos_d_token_type_ids, pos_d_attention_mask, neg_d_input_ids, neg_d_token_type_ids, neg_d_attention_mask):

        # Encode queries
        embed_q = self.bert_q(input_ids=q_input_ids, attention_mask=q_attention_mask, token_type_ids=q_token_type_ids).last_hidden_state[:, 0, :]

        # Encode positive documents
        embed_pos_d = self.bert_d(input_ids=pos_d_input_ids, attention_mask=pos_d_attention_mask, token_type_ids=pos_d_token_type_ids).last_hidden_state[:, 0, :]

        # Encode negative documents
        embed_neg_d = self.bert_d(input_ids=neg_d_input_ids, attention_mask=neg_d_attention_mask, token_type_ids=neg_d_token_type_ids).last_hidden_state[:, 0, :]

        return embed_q, embed_pos_d, embed_neg_d		

    @staticmethod
    def compute_loss(embed_q, embed_pos_d, embed_neg_d, beta, alpha):
        """
        Compute combined loss:
        1. Softmax-normalized in-batch NLL loss
        2. Triplet loss with Euclidean distance
        """
        # Softmax-normalized NLL Loss
        logits = torch.matmul(embed_q, embed_pos_d.T)  # Query to positive similarity
        logits = F.softmax(logits, dim=1)
        labels = torch.arange(logits.size(0)).to(logits.device)  # Diagonal as correct matches
        nll_loss = F.cross_entropy(logits, labels)

        # Triplet Loss (Euclidean Distance)
        pos_dist = torch.norm(embed_q - embed_pos_d, p=2, dim=1)
        neg_dist = torch.norm(embed_q - embed_neg_d, p=2, dim=1)
        triplet_loss = F.relu(pos_dist - neg_dist + alpha).mean()

        # Weighted Combination
        loss = beta * nll_loss + (1 - beta) * triplet_loss
        return loss			  

        # B = embed_q.size(dim=0)
        # qd_scores = torch.matmul(embed_q, torch.transpose(embed_d, 1, 0)) # B x B

        # # q to d softmax
        # q2d_softmax = F.log_softmax(qd_scores, dim=1)

        # # d to q softmax
        # d2q_softmax = F.log_softmax(qd_scores, dim=0)
        
        # # positive indices (diagonal)
        # pos_inds = torch.tensor(list(range(B)), dtype=torch.long).to(self.args.device)
        
        # q2d_loss = F.nll_loss(q2d_softmax,
        # 	pos_inds,
        # 	weight=weights,
        # 	reduction="mean"
        # )

        # d2q_loss = F.nll_loss(d2q_softmax,
        # 	pos_inds,
        # 	weight=weights,
        # 	reduction="mean"
        # )

        # loss = self.args.alpha * q2d_loss + (1 - self.args.alpha) * d2q_loss

        # return loss
