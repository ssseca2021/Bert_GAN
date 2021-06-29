
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy
from tools.finetuning_argparse_eca import get_argparse 
from torch.autograd import Variable
import numpy
from .layers.GAN import GraphAttentionLayer



args = get_argparse().parse_args()
class BertForECA(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction)
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForECA, self).__init__(config)
        self.bert = BertModel(config)
        self.bert_gate = BertModel(config) #用来 编译和情绪连接之后，每个子句的表示
        self.hidden_size = config.hidden_size

         # self.GAN = GraphAttentionLayer()
        self.GAN = torch.nn.ModuleList([GraphAttentionLayer(in_features =  config.hidden_size, out_features = args.hidden_size, dropout =  args.dropout, alpha = args.alpha) for i in range(args.heads)])
        self.clause_out = nn.Linear(config.hidden_size, 1) #将子句的表达降维

        # self.W1 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.W2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.num_outputs = nn.Linear(config.hidden_size, args.cause_num)
        self.cause_num = args.cause_num
        self.Gra = args.Gra

        self.init_weights()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,  context_mask = None, combine_input_ids = None, combine_token_type_ids=None, combine_attention_mask=None,  clause_mask = None, start_positions=None, end_positions=None, testing = None):
        '''
        input_ids:[batch, max_len]
        token_type_ids:[batch, max_len]
        attention_mask:[batch, max_len]
        start_positions:[batch, max_len]
        end_positions:[batch, max_len]
        y_num:[batch]
        '''
        # print('input_ids.size() = ', input_ids.size())
        # print('token_type_ids.size() = ', token_type_ids.size())
        # print('attention_mask.size() = ', attention_mask.size())

        # print('input_ids.size() = ', input_ids)
        # print('token_type_ids.size() = ', token_type_ids)
        # print('attention_mask.size() = ', attention_mask)

        batch_num = input_ids.size(0) #batch number
        all_clause_len = input_ids.size(1) #the lenths of all the clauses

        combine_num = combine_input_ids.size(0) #batch number * max_clause_num
        max_clause_num = int(combine_num / batch_num) #max_clause_num
        max_clause_len = int((all_clause_len -2) / max_clause_num)
        
        sequence_output_combine, _ = self.bert_gate(combine_input_ids, combine_token_type_ids, combine_attention_mask) #[batch, max_len, dim]
        cls_p_combine = sequence_output_combine[:, 0, :] #[batch * max_clause_len, dim]

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask) #[batch, max_len, dim]
        cls_p = sequence_output[:, 0, :] #[batch, dim]

        #获取每个clause的表示
        cls_c = torch.reshape(cls_p_combine, [-1, max_clause_num, self.hidden_size]) #[batch, clause_num, hidden_size]
        
        #构建图
        a1 = cls_c
        a2 = torch.transpose(cls_c, 1,2)
        ATT_M = torch.bmm(a1, a2) #[batch，clause_num, clause_num]
        # print('ATT-M  = ', ATT_M)
    
        att_a = ATT_M.cpu().detach().numpy() #传入子句的真实长度
        clause_weight = numpy.zeros(shape = [batch_num, max_clause_num, max_clause_num])
        for i in range(batch_num):
            # print('context_len = ', context_len)
            for j in range(max_clause_num):
                cutten = att_a[i][j].argsort()
                ll = cutten.shape[0]
                numm = min(max_clause_num, self.Gra)
                for k in range(numm):
                    clause_weight[i][j][cutten[k]] = 1

        ATT_M = torch.tensor(clause_weight).cuda()

        #需要对ATT_M进行归一化和mask
        for item in self.GAN:
            cls_c = item(h = cls_c, adj = ATT_M) #[batch, clause_num, dim]

        cla_att = self.clause_out(cls_c)
        cla_att = cla_att.squeeze(-1) #[batch, clause_num]

        zero_vec = -9e15*torch.ones_like(cla_att) 
        clause_attention = torch.where(clause_mask > 0, cla_att, zero_vec) #这里将填充的clause的位置填充为0
        clause_attention = torch.softmax(clause_attention, -1) #将最后一个维度进行归一化 [batch, clause_num]
        # print('clause_attention = ',clause_attention)

        #扩充attention
        clause_attention = clause_attention.unsqueeze(-1) #[batch, clause_num, 1]
        clause_attention = clause_attention.repeat([1, 1, max_clause_len]) #[batch, clause_num, each_clause_len]
        clause_attention = torch.reshape(clause_attention, (-1, all_clause_len - 2)) #[batch, clause_num * each_clause_len]

        x1 =torch.ones(batch_num,1).cuda()
        a = torch.cat((x1, clause_attention),1) 
        a = torch.cat((a, x1), 1) #[batch, 2 + clause_num * each_clause_len]

        zero_vec_ = -9e15*torch.ones_like(input_ids) #[batch, 1 + clause_num * each_clause_len]
        word_attention = torch.where(context_mask > 0, a, zero_vec_) #[batch, 1 + clause_num * each_clause_len]
        word_attention = torch.softmax(word_attention, -1) #[batch, 1 + clause_num * each_clause_len]

        word_attention = word_attention.unsqueeze(-1) #[batch, 1 + clause_num * each_clause_len, 1]
        word_attention = word_attention.repeat([1, 1, self.hidden_size]) #[batch, 1 + clause_num * each_clause_len, hidden_size]
        sequence_output = sequence_output.mul(word_attention) #[batch, 1 + clause_num * each_clause_len, hidden_size] 将注意力作用于每个词语上

        if args.cause_num != 1:
            logits_num = self.num_outputs(cls_p)

        # print('sequence_output.size = ', sequence_output.size())
        logits = self.qa_outputs(sequence_output) #[batch, max_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  #[batch, max_len]
        end_logits = end_logits.squeeze(-1) # #[batch, max_len]

        logs_fun = torch.nn.LogSoftmax(dim=1)
        batch_size = sequence_output.size(0)
        seq_length = sequence_output.size(1)

        def compute_loss(para_logits, positions):
            log_probs = logs_fun(para_logits)
            loss = - torch.mean(torch.sum(positions * log_probs, dim=-1))
            return loss
        
        def compute_span_num_loss(num_logits, nums):
            # print('a= ', positions.size())
            # print('b = ', para_logits.size())
            one_hot_positions = torch.zeros(batch_size, self.cause_num).cuda()
            one_hot_positions = one_hot_positions.scatter_(1, nums.unsqueeze(1), 1)
            log_probs = logs_fun(num_logits)
            loss = - torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))
            return loss

        if start_positions is not None and end_positions is not None:
            y_num = torch.sum(start_positions, -1) - 1  #[batch]
            # print('y_num= ', y_num)
            assert torch.sum(start_positions) == torch.sum(end_positions)
            # If we are on multi-GPU, split add a dimension
            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if args.cause_num == 1:
                y_num_loss = 0
            else:
                y_num_loss = compute_span_num_loss(logits_num, y_num)

            total_loss += y_num_loss
            return (total_loss,)
        else:
            if args.cause_num == 1:
                return (start_logits, end_logits, [1]*end_logits.size(0))
            else:
                return (start_logits, end_logits, logits_num.argmax(-1))






class BertForECA_Gra(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction)
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForECA_Gra, self).__init__(config)
        self.bert = BertModel(config)
        self.bert_gate = BertModel(config) #用来 编译和情绪连接之后，每个子句的表示
        self.hidden_size = config.hidden_size

         # self.GAN = GraphAttentionLayer()
        self.GAN = torch.nn.ModuleList([GraphAttentionLayer(in_features =  config.hidden_size, out_features = args.hidden_size, dropout =  args.dropout, alpha = args.alpha) for i in range(args.heads)])
        self.clause_out = nn.Linear(config.hidden_size, 1) #将子句的表达降维

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.num_outputs = nn.Linear(config.hidden_size, args.cause_num)
        self.cause_num = args.cause_num
        self.Gra = args.Gra
        self.GraV = args.GraV

        self.init_weights()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,  context_mask = None, combine_input_ids = None, combine_token_type_ids=None, combine_attention_mask=None,  clause_mask = None, start_positions=None, end_positions=None, testing = None):
        '''
        input_ids:[batch, max_len]
        token_type_ids:[batch, max_len]
        attention_mask:[batch, max_len]
        start_positions:[batch, max_len]
        end_positions:[batch, max_len]
        y_num:[batch]
        '''
        # print('input_ids.size() = ', input_ids.size())
        # print('token_type_ids.size() = ', token_type_ids.size())
        # print('attention_mask.size() = ', attention_mask.size())

        # print('input_ids.size() = ', input_ids)
        # print('token_type_ids.size() = ', token_type_ids)
        # print('attention_mask.size() = ', attention_mask)

        batch_num = input_ids.size(0) #batch number
        all_clause_len = input_ids.size(1) #the lenths of all the clauses

        combine_num = combine_input_ids.size(0) #batch number * max_clause_num
        max_clause_num = int(combine_num / batch_num) #max_clause_num
        max_clause_len = int((all_clause_len -2) / max_clause_num)
        
        sequence_output_combine, _ = self.bert_gate(combine_input_ids, combine_token_type_ids, combine_attention_mask) #[batch, max_len, dim]
        cls_p_combine = sequence_output_combine[:, 0, :] #[batch * max_clause_len, dim]

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask) #[batch, max_len, dim]
        cls_p = sequence_output[:, 0, :] #[batch, dim]

        #获取每个clause的表示
        cls_c = torch.reshape(cls_p_combine, [-1, max_clause_num, self.hidden_size]) #[batch, clause_num, hidden_size]
        
        #构建图
        cls_c_T = torch.transpose(cls_c, 1,2)
        ATT_M = torch.bmm(cls_c, cls_c_T) #[batch，clause_num, clause_num]

        # att_a = ATT_M.cpu().detach().numpy() #传入子句的真实长度
        # clause_weight = numpy.zeros(shape = [batch_num, max_clause_num, max_clause_num])
        
        zero_vec = -9e15*torch.ones_like(ATT_M) 
        one_vec =  torch.ones_like(ATT_M) 
        ATT_M = torch.where(ATT_M/1000 > self.GraV, one_vec, zero_vec) #这里将填充的clause的位置填充为0

        # for i in range(batch_num):
        #     # print('context_len = ', context_len)
        #     for j in range(max_clause_num):
        #         cutten = att_a[i][j].argsort()
        #         ll = cutten.shape[0]
        #         numm = min(ll, self.Gra)
        #         for k in range(numm):
        #             clause_weight[i][j][cutten[k]] = 1

        # ATT_M = torch.tensor(clause_weight).cuda()

        #需要对ATT_M进行归一化和mask
        for item in self.GAN:
            cls_c = item(h = cls_c, adj = ATT_M) #[batch, clause_num, dim]

        cla_att = self.clause_out(cls_c)
        cla_att = cla_att.squeeze(-1) #[batch, clause_num]

        zero_vec = -9e15*torch.ones_like(cla_att) 
        clause_attention = torch.where(clause_mask > 0, cla_att, zero_vec) #这里将填充的clause的位置填充为0
        clause_attention = torch.softmax(clause_attention, -1) #将最后一个维度进行归一化 [batch, clause_num]
        print('clause_attention = ',clause_attention)

        #扩充attention
        clause_attention = clause_attention.unsqueeze(-1) #[batch, clause_num, 1]
        clause_attention = clause_attention.repeat([1, 1, max_clause_len]) #[batch, clause_num, each_clause_len]
        clause_attention = torch.reshape(clause_attention, (-1, all_clause_len - 2)) #[batch, clause_num * each_clause_len]

        x1 =torch.ones(batch_num,1).cuda()
        a = torch.cat((x1, clause_attention),1) 
        a = torch.cat((a, x1), 1) #[batch, 2 + clause_num * each_clause_len]

        zero_vec_ = -9e15*torch.ones_like(input_ids) #[batch, 1 + clause_num * each_clause_len]
        word_attention = torch.where(context_mask > 0, a, zero_vec_) #[batch, 1 + clause_num * each_clause_len]
        word_attention = torch.softmax(word_attention, -1) #[batch, 1 + clause_num * each_clause_len]

        word_attention = word_attention.unsqueeze(-1) #[batch, 1 + clause_num * each_clause_len, 1]
        word_attention = word_attention.repeat([1, 1, self.hidden_size]) #[batch, 1 + clause_num * each_clause_len, hidden_size]
        sequence_output = sequence_output.mul(word_attention) #[batch, 1 + clause_num * each_clause_len, hidden_size] 将注意力作用于每个词语上

        if args.cause_num != 1:
            logits_num = self.num_outputs(cls_p)

        # print('sequence_output.size = ', sequence_output.size())
        logits = self.qa_outputs(sequence_output) #[batch, max_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  #[batch, max_len]
        end_logits = end_logits.squeeze(-1) # #[batch, max_len]

        logs_fun = torch.nn.LogSoftmax(dim=1)
        batch_size = sequence_output.size(0)
        seq_length = sequence_output.size(1)

        def compute_loss(para_logits, positions):
            log_probs = logs_fun(para_logits)
            loss = - torch.mean(torch.sum(positions * log_probs, dim=-1))
            return loss
        
        def compute_span_num_loss(num_logits, nums):
            # print('a= ', positions.size())
            # print('b = ', para_logits.size())
            one_hot_positions = torch.zeros(batch_size, self.cause_num).cuda()
            one_hot_positions = one_hot_positions.scatter_(1, nums.unsqueeze(1), 1)
            log_probs = logs_fun(num_logits)
            loss = - torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))
            return loss

        if start_positions is not None and end_positions is not None:
            y_num = torch.sum(start_positions, -1) - 1  #[batch]
            # print('y_num= ', y_num)
            assert torch.sum(start_positions) == torch.sum(end_positions)
            # If we are on multi-GPU, split add a dimension
            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            if args.cause_num == 1:
                y_num_loss = 0
            else:
                y_num_loss = compute_span_num_loss(logits_num, y_num)

            total_loss += y_num_loss
            return (total_loss,)
        else:
            if args.cause_num == 1:
                return (start_logits, end_logits, [1]*end_logits.size(0))
            else:
                return (start_logits, end_logits, logits_num.argmax(-1))
