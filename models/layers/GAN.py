import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):

        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, h, adj):
        #adj.shape:[2708, 2708]
        """
        h:[batch, clause_num, dim]
        adj:[batch, clause_num, clause_num]
        """
        batch = h.size(0) #doc num
        max_clause_num = h.size(1) #clause num
        # print('h.size = ', h.size())
        # print('batch = ', batch)
        # print('max_len = ', max_len)
        # print('self.in_features = ', self.in_features)
        h = h.contiguous().view(batch * max_clause_num, self.in_features)
        Wh = torch.mm(h, self.W) # h.shape: (N= 2708 (节点数), in_features = 1433), W.shape(1433, 8); Wh.shape: (N, out_features)
        Wh = Wh.view(batch, max_clause_num, self.out_features) #[batch, max_clause_num, dim]

        a_input = self._prepare_attentional_mechanism_input(Wh) #[batch, 2708, 2708, 8 * 8]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) #[batch, max_clause_num, max_clause_num] #注意力

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) #这里需要adj填充的时候要填充为0  #[batch, max_len, max_len]
        attention = F.softmax(attention, dim=2) #[batch, max_len, max_len]
        a = torch.zeros_like(attention) 
        attention = torch.where(adj>0, attention, a) #去掉那些填充的元素为0
        # print('attention = ', attention)
        # attention = F.dropout(attention, self.dropout, training=self.training) #[batch, max_len, max_len]
        # print('attention drop = ', attention)
        h_prime = torch.matmul(attention, Wh) #[batch, max_len, dim]
        # print('h_prime = ', h_prime)
        if self.concat:
            # print('F.elu(h_prime) = ', F.elu(h_prime))
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh):
        batch = Wh.size(0)
        N = Wh.size()[1] # number of nodes max_len

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (batch, N * N, out_features)
        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        return all_combinations_matrix.view(batch, N, N, 2 * self.out_features)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


