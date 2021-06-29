import torch
from collections import Counter
# from processors.utils_ner import get_entities
import pickle
import json
import numpy as np
import random
import sys
import os

def get_clause_label(pre_cause_span, tru_cause_span, data_len):

    pre_c = [0] * len(data_len)
    tru_c = [0] * len(data_len)

    clause_index = []
    tag = 0
    for index, item in enumerate(data_len):
        clause_index.append((tag, tag+item))
        tag += item
        
    for item in pre_cause_span:
        ss, ee = item[0], item[1]
        for indexc, itemc in enumerate(clause_index):
            ssc, eec = itemc[0], itemc[1]
            minn = max(ssc, ss)
            maxx = min(ee, eec)
            if maxx - minn > 0:
                pre_c[indexc] = 1
    
    for item in tru_cause_span:
        ss, ee = item[0], item[1]
        for indexc, itemc in enumerate(clause_index):
            ssc, eec = itemc[0], itemc[1]
            minn = max(ssc, ss)
            maxx = min(ee, eec)
            if maxx - minn > 0:
                tru_c[indexc] = 1
    
    trucc = sum(tru_c)
    if trucc <= 0:
        print('tru_cause_span = ', tru_cause_span)
    # assert trucc > 0
    precc = sum(pre_c)
    corrcc = np.sum(np.array(tru_c) * np.array(pre_c))
    return trucc, precc, corrcc


def get_prf(pre_cause_span_indexs, true_cause_span_indexs, clause_lens, examples):
    """
    pre_cause_span_indexs:[batch, sjj]
    true_cause_span_indexs:[batch, sij] 每个元素是一个元组（s,e）
    eval_examples
    """
    # print('len(pre_cause_span_indexs) = ', len(pre_cause_span_indexs))
    # print('len(true_cause_span_indexs) = ', len(true_cause_span_indexs))
    # print('len(examples) = ', len(examples))

    assert len(pre_cause_span_indexs) == len(true_cause_span_indexs) == len(examples)
    batch_num = len(pre_cause_span_indexs)

    # data_len_c = []
    # for index, item in enumerate(examples):
    #     data_len_c.append(item.data_len_c)

    tru_span, pre_span, corr_span = 0,0,0
    tru_cl, pre_cl, corr_cl = 0,0,0

    p_ph, r_ph = 0.0, 0.0

    num_zeros = 0

    for i in range(batch_num):
        tru_ph, pre_ph, corr_ph = 0.0, 0.0, 0.0

        pre_cause_span = pre_cause_span_indexs[i]
        tru_cause_span = true_cause_span_indexs[i]
        exam = examples[i]
        data_len = clause_lens[i] #是一个列表【2，3，2，4】每个数字代表子句的长度
        
        tru_span += len(tru_cause_span)
        pre_span += len(pre_cause_span)

        tru_cl_, pre_cl_, corr_cl_ = get_clause_label(pre_cause_span, tru_cause_span, data_len)
        tru_cl += tru_cl_
        pre_cl += pre_cl_
        corr_cl += corr_cl_

        for itemp in pre_cause_span:
            pre_ph += itemp[1] - itemp[0] + 1
        for itemt in tru_cause_span:
            tru_ph += itemt[1] - itemt[0] + 1
       
        for j in range(len(pre_cause_span)):
            pre = pre_cause_span[j]
            for k in range(len(tru_cause_span)):
                tru = tru_cause_span[k]
                if pre == tru:
                    corr_span += 1
                
                ss = max(pre[0], tru[0])
                ee = min(pre[1], tru[1])
                if ss <= ee:
                    corr_ph +=  ee-ss + 1

        p_ph += 1.0 * corr_ph / pre_ph if pre_ph > 0 else 0
        r_ph += 1.0 * corr_ph / tru_ph if tru_ph > 0 else 0
        if tru_ph ==0:
            num_zeros += 1
    
    print('num_zeros = ', num_zeros)
    print('batch_num = ', batch_num)
    pph = p_ph/batch_num
    rph = r_ph/batch_num
    fph = 2 * pph * rph/ (pph + rph) if (pph + rph) >0 else 0

    pc = 1.0 * corr_cl/pre_cl if pre_cl > 0  else 0
    rc = 1.0 * corr_cl / tru_cl
    fc = 2*pc*rc/(pc+rc) if pc + rc > 0 else 0


    ps = corr_span / pre_span if pre_span > 0  else 0
    rs = corr_span / tru_span
    fs = 2*ps*rs / (ps+rs) if ps +rs > 0 else 0


    result = {'span_p': np.around(ps, decimals=4),'span_r': np.around(rs, decimals=4), 'span_f': np.around(fs, decimals=4), 'w_p': np.around(pph, decimals=4),'w_r': np.around(rph, decimals=4), 'w_f': np.around(fph, decimals=4), 'pc': np.around(pc, decimals=4), 'rc': np.around(rc, decimals=4), 'fc': np.around(fc, decimals=4)}
    result_span = "span: {} {} {}".format(np.around(ps, decimals=4), np.around(rs, decimals=4), np.around(fs, decimals=4))
    result_word = "word: {} {} {}".format(np.around(pph, decimals=4), np.around(rph, decimals=4), np.around(fph, decimals=4))
    result_c = "clause: {} {} {}".format(np.around(pc, decimals=4),  np.around(rc, decimals=4), np.around(fc, decimals=4))
    
    print('\n')
    print(result_span)
    print(result_word)
    print(result_c)

    return result






    

                
                

                







