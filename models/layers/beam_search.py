import torch

def get_cause_span(start_logit, end_logit):
    """
    start_logit:[batch, seqence_len]
    end_logit: [batch, seqence_len]
    """
    start_l, start_indices = torch.sort(start_logit, descending=True) #从大到小进行排序，返回值是值的大小和值的index
    end_l, end_indices = torch.sort(end_logit, descending=True) #从大到小进行排序，返回值是值的大小和值的index
    



