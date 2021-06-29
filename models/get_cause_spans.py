

import torch
def get_pre_indexs(start_logits, end_logits, cause_num, max_num):
    """
    start_logits:[batch, nnn]
    end_logits:[batch, nnn]
    cause_num:[batch]
    """
    spans_index  = []
    for start_l, end_l in zip(start_logits, end_logits):
        spans_index_doc = []
        start_log, start_index  = torch.sort(start_l, descending= True) #从大到小排序
        end_log, end_index  = torch.sort(end_l, descending= True) #从大到小排序

        start_log = start_log.detach().cpu().numpy().tolist()
        start_index = start_index.detach().cpu().numpy().tolist()

        end_log = end_log.detach().cpu().numpy().tolist()
        end_index = end_index.detach().cpu().numpy().tolist()

        endd = -1
        for i in range(max_num):
            for s in start_index:
                if s < endd:
                    continue
                for e in end_index:
                    if s <= e and e > endd:
                        spans_index_doc.append((s,e))
                        endd = e
                        break
        spans_index.append(spans_index_doc)
    
    final_cause_span = []
    for num, span in zip(cause_num, spans_index):
        if len(span) < num + 1:
            final_cause_span.append(span)
        else:
            final_cause_span.append(span[0: num + 1])
    return final_cause_span 


def get_tru_indexs(start_position, end_position):
    """
    start_logits:[batch, nnn]
    end_logits:[batch, nnnn]
    """
    spans_index  = []
    for start_l, end_l in zip(start_position, end_position):
        # print('start_l= ', start_l)
        # print('end_l= ', end_l)
        aa = []
        s_indexs_doc, e_indexs_doc = [], []
        for indexs, items in enumerate(start_l):
            if items == 1:
                s_indexs_doc.append(indexs)

        for indexe, iteme in enumerate(end_l):
            if iteme == 1:
                e_indexs_doc.append(indexe)
                
        for a, b in zip(s_indexs_doc, e_indexs_doc):
            aa.append((a,b))
        # print('end_l = ', sum(end_l))
        spans_index.append(aa)

    return spans_index


def get_pre_sti_indexs(start_logits, end_logits):
    """
    start_logits:[batch, nnn]
    end_logits:[batch, nnn]
    cause_num:[batch]
    max_num:
    """
    spans_index  = []
    for start_l, end_l in zip(start_logits, end_logits):
        spans_index_doc = []
        spans_logits = []
        start_log, start_index  = torch.sort(start_l, descending= True) #从大到小排序
        end_log, end_index  = torch.sort(end_l, descending= True) #从大到小排序

        start_log = start_log.detach().cpu().numpy().tolist()
        start_index = start_index.detach().cpu().numpy().tolist()

        end_log = end_log.detach().cpu().numpy().tolist()
        end_index = end_index.detach().cpu().numpy().tolist()

        for i in range(20):
            for s in start_index:
                for e in end_index:
                    if s < e:
                        spans_logits.append(start_l[s] + end_l[e])
                        spans_index_doc.append((s,e))
                        break

        tagg = 0
        max_log = -100000000000000
        for index, loggits in enumerate(spans_logits):
            if max_log < loggits:
                max_log = loggits
                tagg = index
        spans_index.append([spans_index_doc[tagg]])

    return spans_index 
