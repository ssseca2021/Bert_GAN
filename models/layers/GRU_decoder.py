#encoding=utf-8
import argparse
import torch
import time
import json
import numpy as np
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F


def index(i, max_len):
    if i == 0:
        tmp = [x_ + 2 for x_ in range(max_len)]
    else:
        tmp = [x_ + 1 + 2 for x_ in range(i)][::-1] + [x_ + 2 for x_ in range(max_len - i)] #这个是什么意思？需要bug一下。
    #log2
    tmp = np.log2(tmp)
    tmp = tmp.reshape((1, max_len))
    return torch.from_numpy(tmp).float().cuda()

class Attention(torch.nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_hidden_size):
        super(Attention, self).__init__()
        self.attn = torch.nn.Linear(encoder_hidden_size*2 + decoder_hidden_size, attention_hidden_size)
        # self.attn = torch.nn.Linear(encoder_hidden_size*6 + decoder_hidden_size, attention_hidden_size)
        self.v = torch.nn.Parameter(torch.rand(attention_hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, attention_vector, encoder_outputs, time_step, max_len, mask=None):
        """
        attention_vector:刚开始是句子编码的时候的因层最后生成的向量，后来就是一在decoder时候生成的向量。[batch, hidden]
        encoder_outputs：编码时产生的向量 [max_len, batch, 2*hidden]
        time_step：解码的第几个词语
        """
        timestep = encoder_outputs.size(0) #max_len
        h = attention_vector.repeat(timestep, 1, 1).transpose(0,1) #[batch, max_len, hidden]
        encoder_outputs = encoder_outputs.transpose(0, 1) #【batch, max_len, 2*dim】
        
        score = self.score(h, encoder_outputs)
        weight= index(time_step, max_len)
        score = score / weight
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)
        return F.softmax(score, dim=1).unsqueeze(1)
    
    def score(self, h, encoder_outputs):
        concat = torch.cat([h, encoder_outputs], 2) #将编码时候的向量矩阵和h矩阵向量进行拼接，h是解码时刻产生的向量，然后进行扩充之后的矩阵 
        s = self.attn(concat) #是一个线性转换，变成了attention  vector [batch, max_len.attention_dim]
        s = s.transpose(1,2) # 变成【batch, attention_dim, max_len】
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1) #v:是一个随机产生的300维度的向量，也就是attention维度的向量，这一步是先进行扩充到batch——size [batch, 1, attention dim]
        s = torch.bmm(v, s)
        return s.squeeze(1)


class Decoder(torch.nn.Module):
    def __init__(self, args, num_classes=3, dropout=0.2):
        super(Decoder, self).__init__()
        self.args = args
        
        self.label_embedding = torch.nn.Embedding(num_classes, args.label_embedding_size)
        self.dropout = torch.nn.Dropout(args.dropout)

        self.attention = Attention(args.encoder_hidden_size, args.decoder_hidden_size, args.attention_hidden_size)
        self.rnn = torch.nn.GRU(args.encoder_hidden_size*2 + args.label_embedding_size, args.decoder_hidden_size, args.decoder_num_layers, batch_first=False, bidirectional=False)

        self.hidden2label = torch.nn.Linear(args.decoder_hidden_size, num_classes)

        self.transformer  = torch.nn.Linear(args.encoder_hidden_size*2, args.decoder_hidden_size)
        self.transformer1 = torch.nn.Linear(args.decoder_hidden_size,   args.decoder_hidden_size)
        self.gate = torch.nn.Linear(args.decoder_hidden_size, args.decoder_hidden_size)


    def forward(self, inputs, last_hidden, encoder_outputs, current_encoder_outputs, time_step, max_len, inputs_mask=None):
        """
        inputs: [batch],
        last_hiddeen: [layer, batch, hidden]
        encoder_outputs:[max_len, batch, 2*hidden]
        current_encoder_outputs: [1, batch, 2*hidden]
        time_step:代表解码第time_step个词语
        max_len：句子的最大长度
        """
        embedded = self.label_embedding(inputs).unsqueeze(0) #[batch, label_size]
        embedded = self.dropout(embedded)
        inputs_mask = inputs_mask.transpose(0,1)
        

        attn_weights = self.attention(last_hidden[-1], encoder_outputs, time_step, max_len, inputs_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        context = context.transpose(0,1)

        rnn_inputs = torch.cat([embedded, context], 2)

        output, hidden = self.rnn(rnn_inputs, last_hidden)
        output = output.squeeze(0)
        
        trans = F.relu(self.transformer(current_encoder_outputs.squeeze(0)))
        trans1= F.relu(self.transformer1(output))
        gate  = self.gate(trans+trans1)
        T = torch.sigmoid(gate)
        C = 1 - T

        output = T*trans + C*output

        output = self.hidden2label(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights


# class Seq2Seq(torch.nn.Module):
#     def __init__(self, encoder, Att_layer, decoder, args):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.Att_layer = Att_layer
#         self.decoder = decoder
#         self.args = args

#     def forward(self, source, source_length, source_e, source_e_length, target=None, testing=False):
#         source_mask = source > 0 #score: 原始数据，是词语的index [8,83] source_mask:tru false的数组 组成的【true， false】
#         source = source.transpose(0, 1) #原始数据变成了【83， 8】
#         source_e_mask = source_e > 0 #score: 原始数据，是词语的index [8,83] source_mask:tru false的数组 组成的【true， false】
#         source_e = source_e.transpose(0, 1) #原始数据变成了【83， 8】

#         target_= target # 是一个PackedSequence的数据
#         max_len = source.size(0) #in other sq2seq, max_len should be target.size() batch——size
#         batch_size = source.size(1) 

#         if target != None:
#             target, _ = torch.nn.utils.rnn.pad_packed_sequence(target, total_length=max_len) #target [83, 8]
        
#         label_size = self.args.label_size
#         outputs =  Variable(torch.zeros(max_len, batch_size, label_size)).cuda()
#         attention = Variable(torch.zeros(max_len, batch_size, max_len)).cuda()

#         # print(source.size(), source_length.size())
#         encoder_outputs, hidden,  encoder_e_outputs, hidden_e= self.encoder(source, source_length, source_e, source_e_length)  # [max_len, batch, 2*hidden_size]; torch.Size([2*layer, batch, hiddendim]) 
#         hidden = hidden[:self.args.decoder_num_layers] # [args.decoder_num_layers, batch, hiddendim]
#         hidden_e = hidden_e[:self.args.decoder_num_layers] # [args.decoder_num_layers, batch, hiddendim]

#         if True:
#             encoder_outputs, hidden = self.Att_layer(source_mask, source_e_mask, encoder_outputs, 
#                             encoder_e_outputs, hidden, hidden_e)
                        

#         output = Variable(torch.zeros((batch_size))).long().cuda()
#         for t in range(max_len):

#             current_encoder_outputs = encoder_outputs[t,:,:].unsqueeze(0) #[1, batch, 2*encoder_hidden_size] 第t个词语的表示，去掉第一个维度，【batch, 2*dim】
#             output, hidden, attn_weights = self.decoder(output, hidden, encoder_outputs, current_encoder_outputs, t, max_len, source_mask)
#             outputs[t] = output
#             attention[t] = attn_weights.squeeze()
#             #is_teacher = random.random() < teacher_forcing_ratio
#             top1 = output.data.max(1)[1]
#             if testing:
#                 output = Variable(top1).cuda()
#             else:
#                 output = Variable(target[t]).cuda()

#         if testing:
#             outputs = outputs.transpose(0,1)
#             return outputs, attention
#         else:
#             packed_y = torch.nn.utils.rnn.pack_padded_sequence(outputs, source_length)
#             score  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y.data), target_.data)
#             return score















